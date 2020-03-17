#include "objreader.hpp"

Utils::ObjReaderRet Utils::readObj(QString filePath)
{
	QVector<QVector3D> rawVertices, rawNormals;
	QVector<QVector2D> rawTextures;
	QVector<int> vertexIndices, textureIndices, normalIndices;

	QVector<Vertex> vertices;
	QVector<GLsizei> indices;

	qDebug() << "engine::ObjLoader:: Reading object: " << filePath;

	QFile obj_file(filePath);
	obj_file.open(QIODevice::ReadOnly);

	QTextStream textStream(&obj_file);

	if (obj_file.isOpen()) {
		while (!textStream.atEnd()) {
			QString line = textStream.readLine();
			QStringList list = line.split(" ");
			bool isTexture = list.count() == 4 ? false : true;
			if (list[0] == "v" && !isTexture) {
				QVector3D vertex;
				for (int i = 1; i < list.count(); i++) {
					vertex[i - 1] = list[i].toFloat();
				}
				//qDebug() << vertex;
				rawVertices.push_back(vertex);
			}
			else if (list[0] == "vt" && isTexture) {
				QVector2D texture;
				for (int i = 1; i < list.count(); i++) {
					texture[i - 1] = list[i].toFloat();
				}
				rawTextures.push_back(texture);
			}
			else if (list[0] == "vn" && !isTexture) {
				QVector3D normal;
				for (int i = 1; i < list.count(); i++) {
					normal[i - 1] = list[i].toFloat();
				}
				//qDebug() << normal;
				rawNormals.push_back(normal);
			}
			else if (list[0] == "f" && !isTexture) {
				// f 1/1/1 2/2/2 3/3/3
				for (int i = 1; i < list.count(); i++) {
					QStringList indexGroup = list[i].split("/");
					vertexIndices.push_back(indexGroup[0].toInt());
					textureIndices.push_back(indexGroup[1].toInt());
					normalIndices.push_back(indexGroup[2].toInt());
				}
				//qDebug() << list;
			}

		}
	}

	for (int i = 0; i < vertexIndices.size(); i++) {
		vertices.push_back(
			Vertex{
				rawVertices[vertexIndices[i] - 1],
				rawNormals[normalIndices[i] - 1],
				rawTextures[textureIndices[i] - 1],
				QVector3D(1.0f, 1.0f, 1.0f),
				QVector3D(1.0f, 1.0f, 1.0f),
				QVector3D(1.0f, 1.0f, 1.0f),
			});
		indices.push_back(i);
	}
	return Utils::ObjReaderRet{ vertices, indices };
}
