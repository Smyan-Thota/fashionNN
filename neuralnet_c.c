#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
 * This program demonstrates a simple feed-forward neural network in C
 * with one hidden layer (64 neurons, ReLU) and an output layer (10 neurons, softmax).
 *
 * It loads the Fashion MNIST dataset from two CSV files:
 *   - fashion_mnist_train.csv
 *   - fashion_mnist_test.csv
 *
 * Each CSV file is assumed to have 785 columns:
 *   pixel0, pixel1, ..., pixel783, label
 *
 * The program performs training (SGD on each example) for a fixed number
 * of epochs, and then evaluates on the test set.
 */

/* ------------------ Hyperparameters ------------------ */
#define EPOCHS        10
/* We'll no longer define LEARNING_RATE here. Instead we use a global variable. */
#define INITIAL_LR    0.0005f

/* The images are 28x28 = 784 pixels. */
#define INPUT_SIZE    784
#define HIDDEN_SIZE   64
#define OUTPUT_SIZE   10

/* Define maximum possible samples to store.
   Adjust if you want to handle more or fewer rows. */
#define MAX_TRAIN_SAMPLES 70000
#define MAX_TEST_SAMPLES  70000

/* ------------------ Global Arrays for data ------------------ */
static float train_images[MAX_TRAIN_SAMPLES][INPUT_SIZE];
static int   train_labels[MAX_TRAIN_SAMPLES];
static int   train_count = 0;  // Actual number of train samples loaded

static float test_images[MAX_TEST_SAMPLES][INPUT_SIZE];
static int   test_labels[MAX_TEST_SAMPLES];
static int   test_count = 0;   // Actual number of test samples loaded

/* ------------------ Global Neural Network Parameters ------------------ */
/* We keep them as 'static' so they're not accessible outside this file. */
static float W1[HIDDEN_SIZE][INPUT_SIZE];  /* Weights for hidden layer */
static float b1[HIDDEN_SIZE];              /* Biases for hidden layer */
static float W2[OUTPUT_SIZE][HIDDEN_SIZE]; /* Weights for output layer */
static float b2[OUTPUT_SIZE];              /* Biases for output layer */

/* ------------------ Current learning rate (global) ------------------ */
static float current_lr = INITIAL_LR;

/* ------------------ Function: CSV Loader ------------------ */
int load_csv(const char *filename, float images[][INPUT_SIZE], int *labels, int max_rows) {
    /*
     * Loads data from a CSV file with 785 columns:
     *   pixel0, pixel1, ..., pixel783, label
     *
     * images  -> will store pixel data as floating-point
     * labels  -> will store integer labels
     * max_rows -> max number of rows we can store
     *
     * Returns the number of samples loaded.
     */

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open CSV file '%s'\n", filename);
        return 0;
    }

    // Buffer for reading lines
    char line[8192];
    int row_count = 0;

    // Read header line first (and discard)
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        fprintf(stderr, "Error: CSV file '%s' is empty?\n", filename);
        return 0;
    }

    // Now read each line of data
    while (fgets(line, sizeof(line), fp)) {
        if (row_count >= max_rows) {
            // If there's more data than we can store, break
            break;
        }
        // Remove trailing newline
        line[strcspn(line, "\r\n")] = '\0';

        // Tokenize by comma
        char *token = strtok(line, ",");
        int col_idx = 0;

        float pixel_values[INPUT_SIZE];
        int label_value = -1;

        while (token != NULL && col_idx <= INPUT_SIZE) {
            if (col_idx < INPUT_SIZE) {
                // Parse pixel value and normalize
                pixel_values[col_idx] = (float)atof(token) / 255.0f;
            } else {
                // Last column is the label
                label_value = atoi(token);
            }
            token = strtok(NULL, ",");
            col_idx++;
        }

        if (col_idx == INPUT_SIZE + 1) {
            // We have 784 pixels + 1 label
            // Copy to global arrays
            for (int i = 0; i < INPUT_SIZE; i++) {
                images[row_count][i] = pixel_values[i];
            }
            labels[row_count] = label_value;
            row_count++;
        }
    }

    fclose(fp);
    printf("Loaded %d samples from '%s'\n", row_count, filename);
    return row_count;
}

/* ------------------ Initialize Weights and Biases ------------------ */
void init_params() {
    srand((unsigned int)time(NULL));

    /* He initialization for the hidden layer */
    float he_init_W1 = sqrtf(2.0f / (float)INPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            float r = (float)rand() / (float)RAND_MAX; // in [0..1]
            float val = (r * 2.0f - 1.0f) * he_init_W1; // random in [-he_init_W1..he_init_W1]
            W1[i][j] = val;
        }
        b1[i] = 0.0f;
    }

    /* He initialization for the output layer */
    float he_init_W2 = sqrtf(2.0f / (float)HIDDEN_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float r = (float)rand() / (float)RAND_MAX;
            float val = (r * 2.0f - 1.0f) * he_init_W2;
            W2[i][j] = val;
        }
        b2[i] = 0.0f;
    }
}

/* ------------------ Forward Pass ------------------ */
void forward(const float *x, float *hidden, float *output) {
    // hidden layer: ReLU(W1*x + b1)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += W1[i][j] * x[j];
        }
        // ReLU
        hidden[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    // output layer: softmax(W2*hidden + b2)
    float maxLogit = -1e9f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += W2[i][j] * hidden[j];
        }
        output[i] = sum;
        if (sum > maxLogit) {
            maxLogit = sum;
        }
    }

    // Softmax
    float sumExp = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = expf(output[i] - maxLogit);
        sumExp += output[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] /= sumExp;
    }
}

/* ------------------ Backward Pass + Update ------------------ */
void backward_and_update(const float *x, float *hidden, float *output, int label) {
    /*
     * label in [0..9].
     * The derivative wrt output: (output[i] - 1.0 if i==label else 0)
     */
    float dOut[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float y_i = (i == label) ? 1.0f : 0.0f;
        dOut[i] = (output[i] - y_i);
    }

    // Grad for W2, b2
    float dHidden[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        dHidden[j] = 0.0f;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float grad = dOut[i] * hidden[j];
            /* Use current_lr (global) instead of a fixed LEARNING_RATE */
            W2[i][j] -= current_lr * grad;

            // accumulate gradient for hidden layer
            dHidden[j] += dOut[i] * W2[i][j];
        }
        b2[i] -= current_lr * dOut[i];
    }

    // Grad for hidden (ReLU)
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        if (hidden[j] <= 0.0f) {
            dHidden[j] = 0.0f;
        }
    }

    // Grad for W1, b1
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < INPUT_SIZE; k++) {
            float grad = dHidden[j] * x[k];
            W1[j][k] -= current_lr * grad;
        }
        b1[j] -= current_lr * dHidden[j];
    }
}

/* ------------------ Train the Network ------------------ */
void train_network() {
    printf("Training for %d epochs on %d samples...\n", EPOCHS, train_count);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        /* Manual learning rate decay: after epoch 5, halve it */
        if (epoch == 5) {
            current_lr *= 0.5f;
            printf("Reduced learning rate to %.6f at epoch %d\n", current_lr, epoch+1);
        }

        float lossSum = 0.0f;
        int correct   = 0;

        // We'll do a simple SGD on each sample
        for (int i = 0; i < train_count; i++) {
            float hidden[HIDDEN_SIZE];
            float output[OUTPUT_SIZE];

            // Forward pass
            forward(train_images[i], hidden, output);

            // Cross-entropy loss = -log(prob of correct class)
            float p_correct = output[train_labels[i]];
            lossSum -= logf(p_correct);

            // Check prediction
            int predicted = 0;
            float maxVal  = -1e9f;
            for (int c = 0; c < OUTPUT_SIZE; c++) {
                if (output[c] > maxVal) {
                    maxVal = output[c];
                    predicted = c;
                }
            }
            if (predicted == train_labels[i]) {
                correct++;
            }

            // Backprop
            backward_and_update(train_images[i], hidden, output, train_labels[i]);
        }

        float avgLoss  = lossSum / (float)train_count;
        float accuracy = (float)correct / (float)train_count * 100.0f;

        printf("Epoch %d/%d - Loss: %.3f, Accuracy: %.2f%%\n",
               epoch + 1, EPOCHS, avgLoss, accuracy);
    }
    printf("Training complete.\n");
}

/* ------------------ Evaluate the Network ------------------ */
void evaluate_network() {
    printf("\nEvaluating on %d test samples...\n", test_count);
    int correct = 0;
    float lossSum = 0.0f;

    for (int i = 0; i < test_count; i++) {
        float hidden[HIDDEN_SIZE];
        float output[OUTPUT_SIZE];

        forward(test_images[i], hidden, output);

        float p_correct = output[test_labels[i]];
        lossSum -= logf(p_correct);

        // Argmax
        int predicted = 0;
        float maxVal = -1e9f;
        for (int c = 0; c < OUTPUT_SIZE; c++) {
            if (output[c] > maxVal) {
                maxVal = output[c];
                predicted = c;
            }
        }
        if (predicted == test_labels[i]) {
            correct++;
        }
    }

    float avgLoss = lossSum / (float)test_count;
    float accuracy = (float)correct / (float)test_count * 100.0f;

    printf("Test Loss: %.3f, Test Accuracy: %.2f%%\n", avgLoss, accuracy);
}

/* ------------------ Main ------------------ */
int main() {
    // 1) Load CSV data for training
    train_count = load_csv("fashion_mnist_train.csv", train_images, train_labels, MAX_TRAIN_SAMPLES);
    if (train_count == 0) {
        printf("No training data loaded. Exiting.\n");
        return 1;
    }

    // 2) Load CSV data for testing
    test_count = load_csv("fashion_mnist_test.csv", test_images, test_labels, MAX_TEST_SAMPLES);
    if (test_count == 0) {
        printf("No testing data loaded. Exiting.\n");
        return 1;
    }

    // 3) Initialize weights/biases
    init_params();

    // 4) Train the network
    train_network();

    // 5) Evaluate on test set
    evaluate_network();

    // 6) Example: Predict for the first test sample
    {
        float hidden[HIDDEN_SIZE];
        float output[OUTPUT_SIZE];
        forward(test_images[0], hidden, output);

        int predicted = 0;
        float maxVal = -1e9f;
        for (int c = 0; c < OUTPUT_SIZE; c++) {
            if (output[c] > maxVal) {
                maxVal = output[c];
                predicted = c;
            }
        }
        printf("\nSample #0 prediction = %d, actual label = %d\n",
               predicted, test_labels[0]);
    }

    return 0;
}
