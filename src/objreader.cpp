#include "objreader.hpp"

Utils::ObjReaderRet Utils::readObj(QString filePath)
{
	using std::vector;
	
	vector<glm::vec3> rawVertices, rawNormals;
	vector<glm::vec2> rawTextures;
	std::vector<int> vertexIndices, textureIndices, normalIndices;

	vector<Vertex> vertices;
	vector<GLsizei> indices;

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
				glm::vec3 vertex;
				for (int i = 1; i < list.count(); i++) {
					vertex[i - 1] = list[i].toFloat();
				}
				//qDebug() << vertex;
				rawVertices.push_back(vertex);
			}
			else if (list[0] == "vt" && isTexture) {
				glm::vec2 texture;
				for (int i = 1; i < list.count(); i++) {
					texture[i - 1] = list[i].toFloat();
				}
				rawTextures.push_back(texture);
			}
			else if (list[0] == "vn" && !isTexture) {
				glm::vec3 normal;
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
				glm::vec3(1.0f),
				glm::vec3(1.0f, 1.0f, 1.0f),
				glm::vec3(1.0f, 1.0f, 1.0f),
			});
		indices.push_back(i);
	}
	return Utils::ObjReaderRet{ vertices, indices };
}
