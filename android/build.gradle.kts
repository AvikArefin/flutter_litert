import java.net.URI
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinAndroidProjectExtension

group = "com.hugocornellier.flutter_litert"
version = "1.0"

buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.13.2")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.3.20")
    }
}

rootProject.allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
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
    compileSdk = 36
    namespace = "com.hugocornellier.flutter_litert"

    externalNativeBuild {
        cmake {
            path = file("../src/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        minSdk = 21
    }
}

// When a Kotlin compile task exists (KGP on AGP < 9, or built-in Kotlin on
// AGP 8.11+/9), it defaults its JVM target to the JDK version (e.g. 21/22)
// and clashes with the Java 17 target. Pin it to 17. Guarded because on
// AGP 9 without KGP the kotlin extension is not registered.
extensions.findByType<KotlinAndroidProjectExtension>()?.apply {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}

dependencies {
    val litert = "1.4.1"
    add("implementation", "com.google.ai.edge.litert:litert:$litert")
    add("implementation", "com.google.ai.edge.litert:litert-gpu:$litert")
}

// LiteRT Next runtime + GPU accelerator (CompiledModel API). Downloaded at
// build time (not shipped in the pub package) and bundled as jniLibs,
// additively alongside the classic Maven litert artifacts above — the
// Interpreter path is unchanged. The release assets are repackaged from
// google-ai-edge/LiteRT prebuilts at the API-pinned commit shared with the
// iOS binaries (see .github/workflows/build-litert-ios.yml notes); both
// ABIs (arm64-v8a, x86_64) are 16 KB page-size aligned.
val litertJniRelease = "litert-android-v1.0.0"
val litertJniDir = layout.buildDirectory.dir("litert-jni")

val downloadLitertJni = tasks.register("downloadLitertJni") {
    val outDir = litertJniDir.get().asFile
    outputs.dir(outDir)
    doLast {
        val marker = File(outDir, "arm64-v8a/libLiteRt.so")
        if (marker.exists()) return@doLast
        outDir.mkdirs()
        val zip = File(outDir, "litert-android-jni.zip")
        val url = "https://github.com/hugocornellier/flutter_litert/releases/" +
            "download/$litertJniRelease/litert-android-jni.zip"
        logger.lifecycle("[flutter_litert] Downloading LiteRT Next Android JNI libraries...")
        URI(url).toURL().openStream().use { input ->
            zip.outputStream().use { output -> input.copyTo(output) }
        }
        copy {
            from(zipTree(zip))
            into(outDir)
            // CPU runtime only for now. The OpenCL/GL GPU accelerator is in
            // the zip but stays unbundled: when it registers in an
            // environment without working OpenCL (every emulator), even
            // GPU|CPU-fallback compilation fails with a runtime error
            // instead of falling back. Include it once the GPU path is
            // validated on real hardware.
            include("**/libLiteRt.so")
        }
        zip.delete()
        if (!marker.exists()) {
            throw GradleException("[flutter_litert] LiteRT Next JNI download did not produce $marker")
        }
    }
}

android {
    sourceSets {
        getByName("main") {
            jniLibs.srcDir(litertJniDir)
        }
    }
}

tasks.named("preBuild") {
    dependsOn(downloadLitertJni)
}
