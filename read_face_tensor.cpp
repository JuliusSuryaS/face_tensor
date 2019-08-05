#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <GL/glut.h>
#include <GL/GL.h>
#include <math.h>

#define vector_vec3d std::vector<cv::Vec3d>
#define vector_vec3f std::vector<cv::Vec3f>

// Namespaces
using std::cout;
using std::cin;
using std::endl;
using std::ios;
using std::string;

// Function proto
std::vector<float> readBinary(const char *filename);
Eigen::MatrixXf kron(Eigen::MatrixXf, Eigen::MatrixXf);
std::vector<cv::Vec3f> readTriangle(char *filename);
std::vector<cv::Vec3d> matToVec(Eigen::MatrixXf);
void computeFaceVertices(int, int);
void objExporter(char *, std::vector<cv::Vec3f>, std::vector<cv::Vec3d>);
void renderScene(void);
void initScene(int, char **);
void reshape(int, int);
void drawModel();
void drawFace();
void computeNormal();
void initLighting();
void myKeyFunc(int, int, int);
void myMouseFunc(int, int, int, int);
cv::Vec3d normalize_vec(cv::Vec3d);

// GLOBAL Variables for opengl
vector_vec3f normalBuffer;
vector_vec3f face;
vector_vec3d vertices;
float modelrot = 0;
bool rotate_param = true;
bool lefthand = false;
bool righthand = true;
Eigen::MatrixXf core_global;
Eigen::MatrixXf u_id_global;
Eigen::MatrixXf u_exp_global;
int id_ctrl_global = 0;
int exp_ctrl_global = 0;


int id_sz = 0;
int exp_sz = 0;
int core_sz = id_sz * exp_sz;

int main(int argc, char **argv){
	/*
		=================================================
		Instruction
		=================================================
		Change core_vec :
		If using different tensor core, change the file
		and 'id_sz', 'exp_sz', 'core_sz'.
		Others would run fine
		--------------------------------------------------
		Example
		--------------------------------------------------
		std::vector<float> core_vec = readBinary("data/faceWareHouse/C1_100_30.bin");
		id_sz = 100;
		exp_sz = 30;
		core_sz = id_sz * exp_sz;
	*/

	// -----------------------
	// Changeable
	// -----------------------
	id_sz = 70;
	exp_sz = 47;
	core_sz = id_sz * exp_sz;

	// Read binary files for vectors
	std::vector<float> core_vec = readBinary("data/faceWareHouse/C1_70_47.bin");
	std::vector<float> u_id_vec = readBinary("data/faceWareHouse/U2.bin");
	std::vector<float> u_exp_vec = readBinary("data/faceWareHouse/U3.bin");
	//std::vector<cv::Vec3f> face_ = readTriangle("data/faceWareHouse/triangle_FaceWareHouse.txt");

	// Map the vector to thier corresponding size
	Eigen::Map<Eigen::MatrixXf> core(core_vec.data(), 34530, core_sz);
	Eigen::Map<Eigen::MatrixXf> u_id(u_id_vec.data(), 150, 150);
	Eigen::Map<Eigen::MatrixXf> u_exp(u_exp_vec.data(), 47, 47);

	// Operations
	// block(start, end, size, size)
	// Get U_id and U_exp size based on reduced matrix
	// Get any index on start
	Eigen::MatrixXf w_id = u_id.block(50, 0, 1, id_sz);
	Eigen::MatrixXf w_exp = u_exp.block(19, 0, 1, exp_sz);

	Eigen::MatrixXf out_kron = kron(w_exp, w_id);
	Eigen::MatrixXf out_mat = core * out_kron.transpose();
	//std::vector<cv::Vec3d> vertices_ = matToVec(out_mat);

	objExporter("data/testData/test_comp.obj", face_, vertices_);

	cout << "Initializing Display" << endl;

	// Init global
	// Global for changing index on w_id and w_exp
	vertices = vertices_;
	face = face_;
	core_global = core;
	u_id_global = u_id;
	u_exp_global = u_exp;

	computeNormal();
	initScene(argc, argv);
}

void initScene(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("FaceWarehouse");
	glutSpecialFunc(myKeyFunc);
	glutMouseFunc(myMouseFunc);
	initLighting();
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);
	glutReshapeFunc(reshape);
	glutMainLoop();
}

void initLighting() {
	GLfloat mat_specular[] = { 0.50, 0.50, 0.50, 0.50 };
	GLfloat mat_shininess[] = { 0.50 };
	GLfloat light_position[] = { 1.0, 1.0, 100.0, 0.0 };
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);

	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
}

void renderScene(void) {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// Render Face
	drawModel();
	// ==========
	glutSwapBuffers();
}

void drawModel() {
	glPushMatrix();
	glTranslatef(0, 0, -2);
	glScalef(0.3, 0.3, 0.3);
	glRotatef(modelrot, 0, 1, 0);
	drawFace();
	//glutSolidCube(1);
	glPopMatrix();
	if (rotate_param) {
		if (righthand && modelrot <= 90) {
			modelrot += 0.5;
		}
		if (modelrot > 90) {
			righthand = false;
			lefthand = true;
		}
		if (lefthand && -90 <= modelrot) {
			modelrot -= 0.5;
		}
		if (modelrot < -90) {
			righthand = true;
			lefthand = false;
		}
	}

}

void drawFace() {
	glBegin(GL_TRIANGLES);

	for (int i = 0; i < face.size(); i++) {
		int indiceA = face[i](0) - 1;
		int indiceB = face[i](1) - 1;
		int indiceC = face[i](2) - 1;

		glColor3f(0.0, 0.0, 0.0);
		glNormal3f(normalBuffer[indiceA](0), normalBuffer[indiceA](1), normalBuffer[indiceA](2));
		glVertex3f(vertices[indiceA](0), vertices[indiceA](1), vertices[indiceA](2));

		glColor3f(0.0, 0.0, 0.0);
		glNormal3f(normalBuffer[indiceB](0), normalBuffer[indiceB](1), normalBuffer[indiceB](2));
		glVertex3f(vertices[indiceB](0), vertices[indiceB](1), vertices[indiceB](2));

		glColor3f(0.0, 0.0, 0.0);
		glNormal3f(normalBuffer[indiceC](0), normalBuffer[indiceC](1), normalBuffer[indiceC](2));
		glVertex3f(vertices[indiceC](0), vertices[indiceC](1), vertices[indiceC](2));
	}
	glEnd();

}

void reshape(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30, (GLfloat)w / (GLfloat)h, 0.1, 1000.0);
	glMatrixMode(GL_MODELVIEW);
}

void myMouseFunc(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			rotate_param == false ? rotate_param = true : rotate_param = false;
		}
	}
}

void myKeyFunc(int key, int x, int y) {
	switch (key) {
		case GLUT_KEY_RIGHT:
			if (exp_ctrl_global < 45){
				exp_ctrl_global++;
			}
			else {
				exp_ctrl_global = 45;
			}
			// Update Face
			cout << "Changed Face" << endl;
			computeFaceVertices(id_ctrl_global, exp_ctrl_global);
			break;
		case GLUT_KEY_LEFT:
			if (exp_ctrl_global > 0){
				exp_ctrl_global--;
			}
			else {
				exp_ctrl_global = 0;
			}
			// Update Face
			cout << "Changed Face" << endl;
			computeFaceVertices(id_ctrl_global, exp_ctrl_global);
			break;
		case GLUT_KEY_UP:
			if (id_ctrl_global > 0){
				id_ctrl_global--;
			}
			else {
				id_ctrl_global = 0;
			}
			// Update Face
			cout << "Changed Expression" << endl;
			computeFaceVertices(id_ctrl_global, exp_ctrl_global);
			break;
		case GLUT_KEY_DOWN:
			if (id_ctrl_global < 148){
				id_ctrl_global++;
			}
			else {
				id_ctrl_global = 148;
			}
			// Update Face
			computeFaceVertices(id_ctrl_global, exp_ctrl_global);
			cout << "Changed Expression" << endl;
			break;
	}

}

void computeNormal() {
	cv::Vec3d AB, BC, normal;
	int countTriangle[22864];
	normalBuffer.resize(vertices.size());
	std::vector<std::vector<int>> adj_vec(vertices.size());
	vector_vec3d face_normal(face.size());

	for (int i = 0; i < face.size(); i++) {

		int vertIdx1 = face[i](0) - 1;
		int vertIdx2 = face[i](1) - 1;
		int vertIdx3 = face[i](2) - 1;

		auto v1 = vertices[vertIdx1];
		auto v2 = vertices[vertIdx2];
		auto v3 = vertices[vertIdx3];

		// save adjacent vector
		adj_vec[vertIdx1].push_back(i);
		adj_vec[vertIdx2].push_back(i);
		adj_vec[vertIdx3].push_back(i);

		auto cross1 = v2 - v1;
		auto cross2 = v3 - v1;

		float normal_x = cross1[1] * cross2[2] - cross1[2] * cross2[1];
		float normal_y = cross1[2] * cross2[0] - cross1[0] * cross2[2];
		float normal_z = cross1[0] * cross2[1] - cross1[1] * cross2[0];

		cv::Vec3d normal = { normal_x, normal_y, normal_z };
		normal = normalize_vec(normal);
		face_normal[i] = normal;

	}
	for (int i = 0; i < vertices.size(); i++) {
		int num_adj = adj_vec[i].size();
		float xtotal = 0.0f;
		float ytotal = 0.0f;
		float ztotal = 0.0f;

		for (int j = 0; j < num_adj; j++) {
			int face_idx = adj_vec[i].at(j);
			auto norm_vec = face_normal[face_idx];
			xtotal += norm_vec[0];
			ytotal += norm_vec[1];
			ztotal += norm_vec[2];
		}
		cv::Vec3d total_norm = { xtotal, ytotal, ztotal };
		total_norm = normalize_vec(total_norm);
		normalBuffer[i] = total_norm;
	}
}

cv::Vec3d normalize_vec(cv::Vec3d v) {
	cv::Vec3d v_norm = {0.0, 0.0, 0.0};
	float length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	for (int i = 0; i < 3; i++) {
		v_norm[i] = v[i] / length;
	}
	return v_norm;
}

void computeFaceVertices(int id_ctrl, int exp_ctrl) {
	Eigen::MatrixXf w_id = u_id_global.block(id_ctrl, 0, 1, id_sz);
	Eigen::MatrixXf w_exp = u_exp_global.block(exp_ctrl, 0, 1, exp_sz);

	Eigen::MatrixXf out_kron = kron(w_exp, w_id);
	Eigen::MatrixXf out_mat = core_global * out_kron.transpose();
	std::vector<cv::Vec3d> vertices_ = matToVec(out_mat);
	vertices = vertices_;
}

std::vector<float> readBinary(const char * filename) {
	std::vector <float> container;
	char* memblock;
	float *data;
	std::streampos size;
	std::fstream myfile;
	myfile.open(filename, ios::in | ios::ate | ios::binary);
	if (myfile.is_open()) {
		size = myfile.tellg();
		memblock = new char[size];
		myfile.seekg(0, ios::beg);
		myfile.read(memblock, size);
		data = (float*)memblock;
		int vector_sz = size / sizeof(data[0]);
		//cout << vector_sz << endl;
		for (int i = 0; i < vector_sz; i++) {
			container.push_back(data[i]);
		}
		//delete data;
		delete[] memblock;

	}
	myfile.close();
	//cout << "core_size:" << container.size()<<endl;

	return container;
}

Eigen::MatrixXf kron(Eigen::MatrixXf A, Eigen::MatrixXf B) {

	int A_rows, A_cols;
	int B_rows, B_cols;

	A_rows = A.rows();
	A_cols = A.cols();
	B_rows = B.rows();
	B_cols = B.cols();

	Eigen::MatrixXf C(A_rows*B_rows, A_cols*B_cols);

	for (int i = 0; i < A_rows; i++) {
		for (int j = 0; j < A_cols; j++) {
			C.block(i*B_rows, j*B_cols, B_rows, B_cols) = A(i, j) * B;
		}
	}

	return C;
}

std::vector<cv::Vec3f> readTriangle(char *filename){
	int tri_num = 22864;
	std::ifstream fin;
	fin.open(filename);
	string others;
	std::vector<cv::Vec3f>triangles(tri_num);
	for (int j = 0; j <tri_num; j++) {
		fin >> others >> triangles[j](0) >> triangles[j](1) >> triangles[j](2);
	}
	fin.close();
	return triangles;
}

std::vector<cv::Vec3d> matToVec(Eigen::MatrixXf vert) {
	int size = 34530 / 3;
	std::vector<cv::Vec3d> vertices(size);

	for (int i = 0; i < size; i++) {
		vertices[i] = cv::Vec3d(vert(3 * i), vert(3 * i + 1), vert(3 * i + 2));
	}
	return vertices;
}

void objExporter(char* filename, std::vector<cv::Vec3f> face, std::vector<cv::Vec3d> vertices) {
	std::ofstream outfile;
	outfile.open(filename, std::ios::out);
	if (outfile.is_open()) {
		for (int i = 0; i < vertices.size(); i++) {
			outfile << "v" << " ";
			outfile << vertices[i](0) << " ";
			outfile << vertices[i](1) << " ";
			outfile << vertices[i](2) << std::endl;
		}
		for (int j = 0; j < face.size(); j++) {
			outfile << "f" << " ";
			outfile << face[j](0) << " ";
			outfile << face[j](1) << " ";
			outfile << face[j](2) << std::endl;
		}
	}
	outfile.close();
}
