/* Edge Impulse ingestion SDK */
#include <g2_inferencing.h>

/* Includes --------------------------------------------------------------- */
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <WiFi.h>
#include <PubSubClient.h>

Adafruit_MPU6050 mpu;

/* LEDs */
#define LED_PARADO   14
#define LED_ANDANDO  27
#define LED_PULANDO  26

/* WiFi + MQTT ------------------------------------------------------------ */
const char* ssid = "AMF-CORP";
const char* password = "@MF$4515";
const char* mqtt_server = "test.mosquitto.org";

WiFiClient espClient;
PubSubClient client(espClient);

/* Constants -------------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f   // limite em G (caso sensor ultrapasse)

static float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };
static float inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

ei_classifier_smooth_t smooth;

/* --- Axis mapping
   Se seu MPU estiver fisicamente rotacionado em relação ao dataset de treino,
   ajuste aqui. Exemplo padrão (identidade):
     AXIS_MAP = {0,1,2} -> x->x, y->y, z->z
     AXIS_SIGN = {1,1,1} -> nenhum eixo invertido

   Se você coletou com o sensor "virado" troque por exemplo:
     AXIS_MAP = {1,0,2} // troca x<->y
     AXIS_SIGN = {-1,1,1} // inverte x
*/
int AXIS_MAP[3]  = {0, 1, 2}; // índice das entradas [x,y,z]
int AXIS_SIGN[3] = { 1,  1,  1}; // 1 ou -1

/* Offsets calculados na calibração (em g) */
float axis_offset[3] = {0.0f, 0.0f, 0.0f};

/* ----------------------------------------------------------------------- */
float ei_get_sign(float number) {
    return (number >= 0.0f) ? 1.0f : -1.0f;
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Conectando ao MQTT...");
        if (client.connect("ESP32-MPU6050")) {
            Serial.println("conectado!");
        } else {
            Serial.print("falha, rc=");
            Serial.println(client.state());
            delay(2000);
        }
    }
}

/* --- Calibração por média:
   - coleta N amostras (com delay entre elas)
   - calcula média por eixo em G
   - define offset = média - expected
   expected assumes sensor is placed so that gravity is +1g on the Z axis (AXIS expected)
*/
void calibrate_sensor(int samples = 400, int delay_ms = 10) {
    Serial.println("Iniciando calibração: mantenha o dispositivo parado na ORIENTAÇÃO de TREINO");
    double sum[3] = {0.0, 0.0, 0.0};
    sensors_event_t a, g, temp;

    for (int i = 0; i < samples; i++) {
        mpu.getEvent(&a, &g, &temp);
        float gx = a.acceleration.x / CONVERT_G_TO_MS2;
        float gy = a.acceleration.y / CONVERT_G_TO_MS2;
        float gz = a.acceleration.z / CONVERT_G_TO_MS2;

        // aplicar mapeamento/inversão para acumular na ordem lógica [X,Y,Z]
        float inVals[3];
        inVals[0] = AXIS_SIGN[0] * ( (AXIS_MAP[0]==0)? gx : (AXIS_MAP[0]==1? gy: gz) );
        inVals[1] = AXIS_SIGN[1] * ( (AXIS_MAP[1]==0)? gx : (AXIS_MAP[1]==1? gy: gz) );
        inVals[2] = AXIS_SIGN[2] * ( (AXIS_MAP[2]==0)? gx : (AXIS_MAP[2]==1? gy: gz) );

        sum[0] += inVals[0];
        sum[1] += inVals[1];
        sum[2] += inVals[2];

        delay(delay_ms);
    }

    float mean[3];
    for (int i = 0; i < 3; i++) mean[i] = sum[i] / samples;

    // assumimos que quando parado: X ≈ 0, Y ≈ 0, Z ≈ +1g
    float expected[3] = {0.0f, 0.0f, 1.0f};

    for (int i = 0; i < 3; i++) {
        axis_offset[i] = mean[i] - expected[i];
    }

    Serial.println("Calibração completa:");
    Serial.print("Médias (g): "); Serial.print(mean[0], 3); Serial.print(", ");
    Serial.print(mean[1], 3); Serial.print(", "); Serial.println(mean[2], 3);

    Serial.print("Offsets aplicados (g): ");
    Serial.print(axis_offset[0], 6); Serial.print(", ");
    Serial.print(axis_offset[1], 6); Serial.print(", ");
    Serial.println(axis_offset[2], 6);

    Serial.println("Se necessário ajuste AXIS_MAP/AXIS_SIGN e recale novamente (recompile).");
}

void setup()
{
    Serial.begin(115200);
    while (!Serial);

    Serial.println("Edge Impulse Inferencing + MQTT + Calibração");

    /* LEDs */
    pinMode(LED_PARADO, OUTPUT);
    pinMode(LED_ANDANDO, OUTPUT);
    pinMode(LED_PULANDO, OUTPUT);

    /* WiFi */
    WiFi.begin(ssid, password);
    Serial.print("Conectando ao WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi conectado!");

    /* MQTT */
    client.setServer(mqtt_server, 1883);

    /* MPU6050 */
    Wire.begin();
    if (!mpu.begin()) {
        Serial.println("Erro ao iniciar o MPU6050!");
    } else {
        Serial.println("MPU6050 OK");
    }

    // smoothing
    ei_classifier_smooth_init(&smooth, 10, 7, 0.8, 0.3);

    Serial.println("Digite 'c' + Enter para calibrar (coleta ~4s).");
    Serial.println("Ou apenas deixe rodando para ver valores corrigidos e predições.");
}

static bool buffer_ready = false;
static int samples_collected = 0;

void loop()
{
    // checa comandos serial para calibração
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "c" || cmd == "C") {
            calibrate_sensor(400, 10); // ~4s
            // resetar contadores do buffer para garantir janela cheia depois da calibração
            buffer_ready = false;
            samples_collected = 0;
        } else {
            Serial.println("Comando desconhecido. Digite 'c' para calibrar.");
        }
    }

    if (!client.connected()) reconnect();
    client.loop();

    uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

    numpy::roll(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, -3);

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // valores brutos em g
    float rawx = a.acceleration.x / CONVERT_G_TO_MS2;
    float rawy = a.acceleration.y / CONVERT_G_TO_MS2;
    float rawz = a.acceleration.z / CONVERT_G_TO_MS2;

    Serial.print("Raw G: ");
    Serial.print(rawx, 3); Serial.print(", ");
    Serial.print(rawy, 3); Serial.print(", ");
    Serial.println(rawz, 3);

    // aplicar map/sign para obter valores na ordem esperada pelo modelo
    float inVals[3];
    float tmp[3] = { rawx, rawy, rawz };
    for (int i = 0; i < 3; i++) {
        int idx = AXIS_MAP[i]; // 0,1,2
        float v = tmp[idx];
        inVals[i] = AXIS_SIGN[i] * v;
    }

    // subtrair offsets calculados (em g)
    float corr[3];
    for (int i = 0; i < 3; i++) {
        corr[i] = inVals[i] - axis_offset[i];
    }

    Serial.print("Corr (g): ");
    Serial.print(corr[0], 3); Serial.print(", ");
    Serial.print(corr[1], 3); Serial.print(", ");
    Serial.println(corr[2], 3);

    // colocar no buffer (corrigido, em g)
    buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3] = corr[0];
    buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 2] = corr[1];
    buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1] = corr[2];

    // clamp em g
    for (int i = 0; i < 3; i++) {
        float &v = buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3 + i];
        if (fabs(v) > MAX_ACCEPTED_RANGE) {
            v = ei_get_sign(v) * MAX_ACCEPTED_RANGE;
        }
    }

    samples_collected += 3;
    if (samples_collected >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        buffer_ready = true;
    }

    if (buffer_ready) {
        memcpy(inference_buffer, buffer,
               EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * sizeof(float));

        signal_t signal;
        numpy::signal_from_buffer(inference_buffer,
                                  EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

        ei_impulse_result_t result = { 0 };
        run_classifier(&signal, &result, false);

        const char* prediction = ei_classifier_smooth_update(&smooth, &result);

        Serial.print("Predição: ");
        Serial.println(prediction);

        /* MQTT */
        client.publish("esp32/acelerometro/status", prediction);

        /* LEDs */
        digitalWrite(LED_PARADO,   strcmp(prediction, "parado") == 0);
        digitalWrite(LED_ANDANDO,  strcmp(prediction, "andando") == 0);
        digitalWrite(LED_PULANDO,  strcmp(prediction, "pulando") == 0);
    }

    uint64_t time_to_wait = next_tick - micros();
    if (time_to_wait > 0) {
        delay(time_to_wait / 1000);
        delayMicroseconds(time_to_wait % 1000);
    }
}
