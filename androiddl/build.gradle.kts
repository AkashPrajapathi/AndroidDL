plugins {
    id("java-library")
    alias(libs.plugins.jetbrains.kotlin.jvm)
    id("com.vanniktech.maven.publish") version "0.37.0"
    signing
}
java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}
kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
    }
}

mavenPublishing {
    publishToMavenCentral()
    signAllPublications()

    coordinates(
        groupId = "io.github.akashprajapathi",
        artifactId = "androiddl",
        version = "0.1.1"
    )

    pom {
        name = "AndroidDL"
        description = "A CPU-based deep learning library for Kotlin."
        inceptionYear = "2026"

        url = "https://github.com/AkashPrajapathi/AndroidDL.git"

        licenses {
            license {
                name = "Apache License 2.0"
                url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
                distribution = "https://www.apache.org/licenses/LICENSE-2.0.txt"
            }
        }

        developers {
            developer {
                id = "akashprajapathi"
                name = "Akash Prajapathi"
                url = "https://github.com/akashprajapathi"
            }
        }

        scm {
            url = "https://github.com/akashprajapathi/AndroidDL"
            connection = "scm:git:git://github.com/akashprajapathi/AndroidDL.git"
            developerConnection = "scm:git:ssh://git@github.com/akashprajapathi/AndroidDL.git"
        }
    }

    signing {
        val signingKey = providers.gradleProperty("signingInMemoryKey").orNull
        val signingPassword = providers.gradleProperty("signingInMemoryKeyPassword").orNull
        val signingKeyId = providers.gradleProperty("signingInMemoryKeyId").orNull

        useInMemoryPgpKeys(
            signingKeyId,
            signingKey,
            signingPassword
        )
    }
}
