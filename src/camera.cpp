#include "camera.hpp"

Camera::Camera(){
	LOG("Camera::Camera() default c_tor");
}

Camera::Camera(
	glm::vec3 posVector, glm::vec3 frontVector, glm::vec3 upVector, 
	float fov, int fbW, int fbH, float Cnear, float Cfar
): m_posVector(posVector), m_frontVector(frontVector), m_upVector(upVector), m_worldUpVector(upVector), m_fov(fov), m_frameBufferWidth(fbW), m_frameBufferHeight(fbH), m_near(Cnear), m_far(Cfar) {
	LOG("Camera::Camera() overloaded c_tor");

	updateViewMatrix();
	updateProjectionMatrix(m_frameBufferWidth, m_frameBufferHeight, m_fov);
}

void Camera::setSpeed(double _speed) {
	m_speed = _speed;
}


void Camera::updateViewMatrix() {
	m_viewMatrix = glm::lookAt(
		m_posVector, m_posVector + m_frontVector, m_upVector
	);
}


void Camera::updateProjectionMatrix(int _frameBufferWidth, int _frameBufferHeight, float _fov) {
	m_frameBufferWidth = _frameBufferWidth;
	m_frameBufferHeight = _frameBufferHeight;
	m_fov = _fov;

	m_projectionMatrix = glm::perspective(
		glm::radians(m_fov),
		m_frameBufferWidth/(float)m_frameBufferHeight,
		m_near, m_far
	);
}

void Camera::resetProjectionMatrix() {
	m_projectionMatrix = glm::perspective(
		glm::radians(m_fov),
		m_frameBufferWidth/ (float) m_frameBufferHeight,
		m_near, m_far
	);
}

glm::mat4x4 Camera::getProjectionMatrix() const {
	return m_projectionMatrix;
}

glm::mat4x4 Camera::getViewMatrix() const  {
	return m_viewMatrix;
}

void Camera::update(float t_yaw, float t_pitch, float fov) {
	glm::vec3 direction;

	direction.x = cos(glm::radians(t_yaw)) * cos(glm::radians(t_pitch));
	direction.y = sin(glm::radians(t_pitch));
	direction.z = sin(glm::radians(t_yaw)) * cos(glm::radians(t_pitch));
	m_frontVector = glm::normalize(direction);

	m_rightVector = glm::normalize(glm::cross(m_frontVector, m_worldUpVector));
	m_upVector = glm::normalize(glm::cross(m_rightVector, m_frontVector));

	m_fov = fov;

	updateViewMatrix();
	updateProjectionMatrix(m_frameBufferWidth, m_frameBufferHeight, m_fov);

};

void Camera::update(MouseProps props) {
};

void Camera::update(WindowProps props) {
	updateProjectionMatrix(props.fbW, props.fbH, m_fov);
};

Camera::~Camera() {

}