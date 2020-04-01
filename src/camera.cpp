#include "camera.hpp"

Camera::Camera(){}

Camera::Camera(
	glm::vec3 posVector, glm::vec3 frontVector, glm::vec3 upVector, 
	float fov, int fbW, int fbH, float Cnear, float Cfar
): posVector(posVector), frontVector(frontVector), upVector(upVector), fov(fov), frameBufferWidth(fbW), frameBufferHeight(fbH), _near(Cnear), _far(Cfar) {
	
	updateViewMatrix();
	updateProjectionMatrix(fbW, fbH);
}

void Camera::setSpeed(double _speed) {
	speed = _speed;
}

void Camera::processKB(Movement movement, float speed){
	if (movement == W) {
		posVector += speed * frontVector;
	}
	if (movement == S) {
		posVector -= speed * frontVector;
	}
	if (movement == D) {
		posVector += speed * upVector;
	}
	if (movement == A) {
		posVector -= speed * upVector;
	}
}


void Camera::updateViewMatrix() {
	viewMatrix = glm::lookAt(
		posVector, posVector + frontVector, upVector
	);
}


void Camera::updateProjectionMatrix(int _frameBufferWidth, int _frameBufferHeight) {
	frameBufferWidth = _frameBufferWidth;
	frameBufferHeight = _frameBufferHeight;

	projectionMatrix = glm::perspective(
		glm::radians(fov),
		_frameBufferWidth/(float)_frameBufferHeight,
		_near, _far
	);
}

void Camera::resetProjectionMatrix() {
	projectionMatrix = glm::perspective(
		glm::radians(fov),
		frameBufferWidth/ (float) frameBufferHeight,
		_near, _far
	);
}

glm::mat4x4 Camera::getProjectionMatrix() const {
	return projectionMatrix;
}

glm::mat4x4 Camera::getViewMatrix() const  {
	return viewMatrix;
}

void Camera::update() {

	resetProjectionMatrix();

}

Camera::~Camera() {

}