plugins {
    id("java-library")
    alias(libs.plugins.jetbrains.kotlin.jvm)
    id("maven-publish")
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

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])

            groupId = "io.github.akashprajapathi"
            artifactId = "androiddl"
            version = "0.1.1"
        }
    }

    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/akashprajapathi/AndroidDL")
            credentials {
                username = System.getenv("GITHUB_ACTOR") ?: "akashprajapathi"
                password = System.getenv("GITHUB_TOKEN") ?: "" // Or set your Personal Access Token
            }
        }
    }
}