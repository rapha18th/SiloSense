plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.silosense.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.silosense.app"
        minSdk = 26          // ORT Android + AudioRecord features want a reasonably modern API level
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // Both the FP32 and INT8 .onnx models live in assets/ — see
    // ../../NEXT_STEPS.md for why both are shipped (the on-device
    // benchmark compares them; the app itself only needs INT8 for the
    // live demo).
    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.cardview:cardview:1.0.0")

    // ONNX Runtime for Android — same .onnx files already produced by
    // scripts/train.py + scripts/quantize.py, no reconversion to TFLite
    // needed. Supports NNAPI and XNNPACK execution providers on-device;
    // confirm which one actually gets used on the target phone the same
    // way the old benchmark_on_device.py did (print session.getProviders()
    // equivalent — see OrtSession API).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    testImplementation("junit:junit:4.13.2")
}
