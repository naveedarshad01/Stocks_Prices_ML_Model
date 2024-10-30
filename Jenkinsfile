pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("naveedarshad001/ml-app:${env.BUILD_NUMBER}")
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub') {
                        docker.image("naveedarshad001/ml-app:${env.BUILD_NUMBER}").push()
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                sh "docker run -d -p 5000:5000 naveedarshad001/ml-app:${env.BUILD_NUMBER}"
            }
        }
    }
}
