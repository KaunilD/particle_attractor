#ifndef OBJREADER_H
#define OBJREADER_H

#include "libs.hpp"
#include "vertex.hpp"
namespace Utils {
	struct ObjReaderRet {
		std::vector<Vertex> v;
		std::vector<GLsizei> i;
	};

	ObjReaderRet readObj(QString filePath);
};
#endif OBJREADER_H