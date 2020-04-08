#include "objreader.hpp"

Utils::ObjReaderRet Utils::readObj(const char * filePath)
{
	using std::vector;
	
	vector<glm::vec3> rawVertices, rawNormals;
	vector<glm::vec2> rawTextures;
	std::vector<int> vertexIndices, textureIndices, normalIndices;

	vector<Vertex> vertices;
	vector<GLsizei> indices;

	LOG("ObjLoader:: Reading object");

	std::ifstream objFile;
	std::string line;
	
	objFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		objFile.open(filePath);
	}
	catch (std::ifstream::failure e)
	{
		LOG("ERROR::ObjLoader::FILE_NOT_SUCCESFULLY_READ");
	}

	while (!objFile.eof()) {
		std::getline(objFile, line);
		std::istringstream ss(line);
		std::string token;

		std::getline(ss, token, ' ');
		//std::cout << line << std::endl;

		if (token == "v") {
			glm::vec3 vertex;
			for (int i = 0; i < 3; i++) {
				std::getline(ss, token, ' ');
				//std::cout << token << std::endl;
				vertex[i] = std::stof(token);
			}
			rawVertices.push_back(vertex);
		}
		else if (token == "vt") {
			glm::vec2 texture;
			for (int i = 0; i < 2; i++) {
				std::getline(ss, token, ' ');
				texture[i] = std::stof(token);
			}
			rawTextures.push_back(texture);
		}
		else if (token == "vn") {
			glm::vec3 normal;
			for (int i = 0; i < 3; i++)
			{
				std::getline(ss, token, ' ');
				normal[i] = std::stof(token);
			}
			rawNormals.push_back(normal);
		}
		else if (token == "f") {
			// f 1/1/1 2/2/2 3/3/3
			for (int i = 0; i < 3; i++) {
				std::getline(ss, token, ' ');
				std::istringstream ss2(token);
				std::string index;

				std::getline(ss2, index, '/');
				vertexIndices.push_back(std::stoul(index));

				std::getline(ss2, index, '/');
				textureIndices.push_back(std::stoul(index));

				std::getline(ss2, index, '/');
				normalIndices.push_back(std::stoul(index));
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
				glm::vec3(1.0f),
				glm::vec3(1.0f),
			});
		indices.push_back(i);
	}
	LOG(rawVertices.size());
	return Utils::ObjReaderRet{ std::move(vertices), std::move(indices) };
}
