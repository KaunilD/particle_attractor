#include <QVector3D>
#include <QVector2D>

#ifndef VERTEX_H
#define VERTEX_H

struct Vertex {
	QVector3D position; // 12bytes
	QVector3D normals;
	QVector2D texture;
	QVector3D color; // 12 bytes
	QVector3D tangent;
	QVector3D bitTangent;
};
#endif