#ifndef CAMERA_H
#define CAMERA_H

#include "libs.hpp"

enum Movement {
	W,
	A,
	S,
	D
};

class Camera {
public:
	float fov;
	float _near;
	float _far;

	int frameBufferWidth, frameBufferHeight;

	double speed;

	QMatrix4x4 *projectionMatrix, *viewMatrix;

	QVector3D posVector, upVector, frontVector;

	Camera();
	Camera(
		QVector3D posVector, QVector3D frontVector, QVector3D upVector, 
		float fov, int fbW, int fbH, float _near, float _far
	);
	~Camera();

	void setSpeed(double speed);
	void processKB(Movement mov, float speed);

	void updateViewMatrix();
	void updateProjectionMatrix(int frameBufferWidth, int frameBufferHeight);
	void resetProjectionMatrix();

	QMatrix4x4 getProjectionMatrix() const;
	QMatrix4x4 getViewMatrix() const;

	void update(QWheelEvent *);
};

#endif