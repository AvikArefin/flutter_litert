import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinAndroidProjectExtension

plugins {
    id("com.android.application")
    id("dev.flutter.flutter-gradle-plugin")
}

val agpVersion = com.android.Version.ANDROID_GRADLE_PLUGIN_VERSION
    .substringBefore('.')
    .toInt()
val builtInKotlinProperty = providers.gradleProperty("android.builtInKotlin").orNull
val isBuiltInKotlinEnabled = agpVersion >= 9 && (builtInKotlinProperty == null || builtInKotlinProperty.toBoolean())
val shouldApplyKotlinAndroidPlugin = agpVersion < 9 || !isBuiltInKotlinEnabled

if (shouldApplyKotlinAndroidPlugin) {
    apply(plugin = "org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.flutter_litert_example"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.example.flutter_litert_example"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // Use debug signing so local release runs work without a keystore.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}

extensions.findByType<KotlinAndroidProjectExtension>()?.apply {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}
