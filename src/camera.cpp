#include "camera.hpp"

Camera::Camera() {
	LOG("Camera::Camera() default c_tor");
}

Camera::Camera(
	glm::vec3 posVector, glm::vec3 frontVector, glm::vec3 upVector,
	float fov, int fbH, int fbW, float Cnear, float Cfar
) : m_posVector(posVector), m_frontVector(frontVector), m_upVector(upVector), m_worldUpVector(upVector), m_fov(fov), m_frameBufferWidth(fbW), m_frameBufferHeight(fbH), m_near(Cnear), m_far(Cfar) {
	LOG("Camera::Camera() overloaded c_tor");

	lastX = fbW / 2.0f;
	lastY = fbH / 2.0f;

	updateViewMatrix();
	updateProjectionMatrix(m_frameBufferWidth, m_frameBufferHeight, m_fov);

	// allocate space for viewProjection matrices in
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
		1.0f,m_near, m_far
	);
}

void Camera::resetProjectionMatrix() {
	m_projectionMatrix = glm::perspective(
		glm::radians(m_fov),
		(float)m_frameBufferWidth / (float)m_frameBufferHeight,
		m_near, m_far
	);
}

glm::mat4x4 Camera::getProjectionMatrix() const {
	return m_projectionMatrix;
}

glm::mat4x4 Camera::getViewMatrix() const {
	return m_viewMatrix;
}

void Camera::update(MouseProps props) {
	if (m_fov >= 1.0f && m_fov <= 45.0f)
		m_fov -= props.yoff;
	else if (m_fov < 1.0f)
		m_fov = 1.0f;
	else if (m_fov > 45.0f)
		m_fov = 45.0f;
	
	updateProjectionMatrix(m_frameBufferWidth, m_frameBufferHeight, m_fov);

	if (m_firstMouse)
	{
		lastX = props.xpos;
		lastY = props.ypos;
		m_firstMouse = false;
	}
	xoffset = props.xpos - lastX;
	yoffset = lastY - props.ypos;
	lastX = props.xpos;
	lastY = props.ypos;

	xoffset *= m_speed;
	yoffset *= m_speed;

	m_yaw += xoffset;
	m_pitch += yoffset;

	if (m_pitch > 89.0f)
		m_pitch = 89.0f;
	if (m_pitch < -89.0f)
		m_pitch = -89.0f;

	glm::vec3 direction;
	direction.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
	direction.y = sin(glm::radians(m_pitch));
	direction.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));

	m_frontVector = glm::normalize(direction);
	
	//std::cout << m_frontVector.r << " " << m_frontVector.g << " " << m_frontVector.b << std::endl;
	//updateViewMatrix();
};

void Camera::update(WindowProps props) {
	updateProjectionMatrix(props.fbW, props.fbH, m_fov);
};

Camera::~Camera() {

}