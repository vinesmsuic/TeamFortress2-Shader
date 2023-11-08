/**
 * @file cs488.h
 * @brief CS488/688 Project code.
 *
 * base code written by Toshiya Hachisuka https://cs.uwaterloo.ca/~thachisu/
 * refractored, modified, and extended by Max Ku m3ku@uwaterloo.ca
 */
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// linear algebra 
#include "linalg.h"
using namespace linalg::aliases;


// animated GIF writer
#include "gif.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>


// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-5f;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);
constexpr int globalNumParticles = 300;


// dynamic camera parameters
float3 globalEye = float3(0.0f, 0.0f, 1.5f);
float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing


// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;


// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();



// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}



// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;
    bool loaded = false;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
        this->loaded = false;
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
        this->loaded = true;
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;

// Environment Map
static Image EnvMap;


// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouseLeftPressed = true;
            double xpos, ypos;
            //getting cursor position
            glfwGetCursorPos(window, &xpos, &ypos);
            printf("mouse x: %f  ", xpos);
            printf("mouse y: %f \n", ypos);
        } else if (action == GLFW_RELEASE) {
            mouseLeftPressed = false;
            if (globalRenderType == RENDER_RAYTRACE) {
                AccumulationBuffer.clear();
                sampleCount = 0;
            }
        }
    }
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A2, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS,
    MAT_PHONG,
    MAT_HALF_LAMBERT,
    MAT_FRESNEL,
    MAT_TOON,
    MAT_XTOON,
    MAT_TF2,
    MAT_TF2_INDEPENDENT,
    MAT_TF2_DEPENDENT
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture K_d
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;

    // New members for specular texture (map_Ks), Normal/bump map
    bool isTexturedKs = false;
    unsigned char* textureKs = nullptr;
    int textureKsWidth = 0;
    int textureKsHeight = 0;

    // New member for glossiness texture (map_Ns), roughness/specular map
    bool isTexturedNs = false;
    unsigned char* textureNs = nullptr;
    int textureNsWidth = 0;
    int textureNsHeight = 0;

    // New member for RampMap (grayscale)
    bool isTexturedAmbient = false;
    unsigned char* textureAmbient = nullptr;
    int textureAmbientWidth = 0;
    int textureAmbientHeight = 0;

    // New member for RampMap (RGB)
    bool isTexturedTF = false;
    unsigned char* textureTF = nullptr;
    int textureTFWidth = 0;
    int textureTFHeight = 0;

	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
	}

    float3 fetchTexture(const float2& tex) const {
        // repeating
        int x = int(tex.x * textureWidth) % textureWidth;
        int y = int(tex.y * textureHeight) % textureHeight;
        if (x < 0) x += textureWidth;
        if (y < 0) y += textureHeight;

        int pix = (x + y * textureWidth) * 3;
        const unsigned char r = texture[pix + 0];
        const unsigned char g = texture[pix + 1];
        const unsigned char b = texture[pix + 2];
        return float3(r, g, b) / 255.0f;
    }

	float3 fetchTextureKs(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureKsWidth) % textureKsWidth;
		int y = int(tex.y * textureKsHeight) % textureKsHeight;
		if (x < 0) x += textureKsWidth;
		if (y < 0) y += textureKsHeight;

        int pix = (x + y * textureKsWidth) * 3;
        const unsigned char r = textureKs[pix + 0];
        const unsigned char g = textureKs[pix + 1];
        const unsigned char b = textureKs[pix + 2];
        return float3(r, g, b) / 255.0f * float3(2.0f) - float3(1.0);
	}

    float fetchTextureNs(const float2& tex) const {
        // repeating
        int x = int(tex.x * textureNsWidth) % textureNsWidth;
        int y = int(tex.y * textureNsHeight) % textureNsHeight;
        if (x < 0) x += textureNsWidth;
        if (y < 0) y += textureNsHeight;

        int pix = (x + y * textureNsWidth) * 1;
        const unsigned char gray = textureNs[pix + 0];
        return float(gray) / 255.0f;
    }

    float3 fetchTextureAmbient(const float2& tex) const {
        // repeating
        int x = int(tex.x * textureAmbientWidth) % textureAmbientWidth;
        int y = int(tex.y * textureAmbientHeight) % textureAmbientHeight;
        if (x < 0) x += textureAmbientWidth;
        if (y < 0) y += textureAmbientHeight;

        int pix = (x + y * textureAmbientWidth) * 3;
        const unsigned char r = textureAmbient[pix + 0];
        const unsigned char g = textureAmbient[pix + 1];
        const unsigned char b = textureAmbient[pix + 2];
        return float3(r, g, b) / 255.0f;
    }

    float3 fetchTextureAmbient1DLookUp(const float& scalar) const {
        int x = int(scalar * textureAmbientWidth) % textureAmbientWidth;
        int y = int(textureAmbientHeight/2); //Height is not considered
        if (x < 0) x += textureAmbientWidth;
        if (y < 0) y += textureAmbientHeight;

        int pix = (x + y * textureAmbientWidth) * 3;
        const unsigned char r = textureAmbient[pix + 0];
        const unsigned char g = textureAmbient[pix + 1];
        const unsigned char b = textureAmbient[pix + 2];
        return float3(r, g, b) / 255.0f;
    }

    float3 fetchTextureTF(const float2& tex) const {
        // repeating
        int x = int(tex.x * textureTFWidth) % textureTFWidth;
        int y = int(tex.y * textureTFHeight) % textureTFHeight;
        if (x < 0) x += textureTFWidth;
        if (y < 0) y += textureTFHeight;

        int pix = (x + y * textureTFWidth) * 3;
        const unsigned char r = textureTF[pix + 0];
        const unsigned char g = textureTF[pix + 1];
        const unsigned char b = textureTF[pix + 2];
        return float3(r, g, b) / 255.0f;
    }

    float3 fetchTextureTF1DLookUp(const float& scalar) const {
        int x = int(scalar * textureTFWidth) % textureTFWidth;
        int y = int(textureTFHeight/2); //Height is not considered
        if (x < 0) x += textureTFWidth;
        if (y < 0) y += textureTFHeight;

        int pix = (x + y * textureTFWidth) * 3;
        const unsigned char r = textureTF[pix + 0];
        const unsigned char g = textureTF[pix + 1];
        const unsigned char b = textureTF[pix + 2];
        return float3(r, g, b) / 255.0f;
    }

    float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};





class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	const Material* material; // const pointer to the material of the intersected object
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() { return minp; };
	float3 get_maxp() { return maxp; };
	float3 get_size() { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
};

// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];
				// not doing anything right now
			}
		}
	}

	// ============================================================================================================
	// Helpers for rasterizeTriangle
    bool checkIsInside(const float4 triPosScreen[3], const float P_x, const float P_y) const{
        // Line Equation Method
        // triPosScreen: tri.positions in Screen space
        // P_x: x in Screen
        // P_y: y in Screen
        float L0 = -(P_x - triPosScreen[0].x)*(triPosScreen[1].y - triPosScreen[0].y) + (P_y - triPosScreen[0].y)*(triPosScreen[1].x - triPosScreen[0].x);
        float L1 = -(P_x - triPosScreen[1].x)*(triPosScreen[2].y - triPosScreen[1].y) + (P_y - triPosScreen[1].y)*(triPosScreen[2].x - triPosScreen[1].x);
        float L2 = -(P_x - triPosScreen[2].x)*(triPosScreen[0].y - triPosScreen[2].y) + (P_y - triPosScreen[2].y)*(triPosScreen[0].x - triPosScreen[2].x);
        if ((L0 > 0) && (L1 > 0) && (L2 > 0))
        {return true;}
        else
        {return false;}
    }

    float3 computeBarycentricCoordinates2D(const float2 P, const float2 A, const float2 B, const float2 C) const{
        //http://courses.cms.caltech.edu/cs171/assignments/hw2/hw2-notes/notes-hw2.html
        auto fij = [](float2 P, float2 i, float2 j)
        {
            float f = (i.y - j.y)*P.x + (j.x - i.x)*P.y + i.x*j.y - j.x*i.y;
            return f;
        };

        float alpha = fij(P, B, C) / fij(A, B, C);
        float beta = fij(P, A, C) / fij(B, A, C);
        float gamma = fij(P, A, B) / fij(C, A, B);
        return {alpha, beta, gamma};
    }

    float4 ndc2screen(const float4 homoPos) const{
        //ndc is -1 to 1
        float x = linalg::lerp(0.0f, globalWidth, (homoPos.x + 1.0f) * 0.5f);
        float y = linalg::lerp(0.0f, globalHeight, (homoPos.y + 1.0f) * 0.5f);
        float z = linalg::lerp(globalDepthMin, globalDepthMax, (homoPos.z + 1.0f) * 0.5f);
        float w = homoPos.w;
        float4 screenPos = {x, y, z, w};
        return screenPos;
    }

	float getInterpolatedAttribute(float3 barycentricCoords, float3 attribute) const{
		//Interpolated Point using barycentricCoords
        return barycentricCoords.x*attribute.x + barycentricCoords.y*attribute.y + barycentricCoords.z*attribute.z;
    }
	// ============================================================================================================

	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		// rasterization of a triangle
		// "plm" should be a matrix that contains perspective projection and the camera matrix
		// you do not need to implement clipping
		// you may call the "shade" function to get the pixel value
		// (you may ignore viewDir for now)
        float4 homoPos[3];
        float4 triPosScreen[3];
        for (int i = 0; i < 3; i++) {
            homoPos[i] = {tri.positions[i].x, tri.positions[i].y, tri.positions[i].z, 1};
            homoPos[i] = mul(plm, homoPos[i]);
            // Keep the w instead of divide it.. we need to use it later.
            homoPos[i] = {homoPos[i].x/homoPos[i].w, homoPos[i].y/homoPos[i].w, homoPos[i].z/homoPos[i].w, homoPos[i].w};
            triPosScreen[i] = ndc2screen({homoPos[i].x, homoPos[i].y, homoPos[i].z, homoPos[i].w});
			
        }
        HitInfo info;
        info.material = &materials[tri.idMaterial];

        // Find the bounding box coordinates
        int minX = std::max(0, (int)(std::floor(std::min(std::min(triPosScreen[0].x, triPosScreen[1].x), triPosScreen[2].x))));
        int maxX = std::min(globalWidth - 1, (int)(std::ceil(std::max(std::max(triPosScreen[0].x, triPosScreen[1].x), triPosScreen[2].x))));
        int minY = std::max(0, (int)(std::floor(std::min(std::min(triPosScreen[0].y, triPosScreen[1].y), triPosScreen[2].y))));
        int maxY = std::min(globalHeight - 1, (int)(std::ceil(std::max(std::max(triPosScreen[0].y, triPosScreen[1].y), triPosScreen[2].y))));

        for (int j = minY; j <= maxY; j++) {
            for (int i = minX; i <= maxX; i++) {
                float P_x = i+0.5;
                float P_y = j+0.5;
                if (this->checkIsInside(triPosScreen, P_x, P_y))
                {
                    float3 barycentricCoords = computeBarycentricCoordinates2D({P_x, P_y}, {triPosScreen[0].x, triPosScreen[0].y}, {triPosScreen[1].x, triPosScreen[1].y}, {triPosScreen[2].x, triPosScreen[2].y});
                    float depth = getInterpolatedAttribute(barycentricCoords, {triPosScreen[0].z, triPosScreen[1].z, triPosScreen[2].z});
                    if (depth < FrameBuffer.depth(i, j))
                    {
                        float InterpolatedW = barycentricCoords.x/triPosScreen[0].w + barycentricCoords.y/triPosScreen[1].w + barycentricCoords.z/triPosScreen[2].w;
                        float InterpolatedP_x = getInterpolatedAttribute(barycentricCoords,
                                                                {tri.texcoords[0].x/triPosScreen[0].w, 
																 tri.texcoords[1].x/triPosScreen[1].w, 
																 tri.texcoords[2].x/triPosScreen[2].w});
                        float InterpolatedP_y = getInterpolatedAttribute(barycentricCoords,
                                                                {tri.texcoords[0].y/triPosScreen[0].w, 
																 tri.texcoords[1].y/triPosScreen[1].w, 
																 tri.texcoords[2].y/triPosScreen[2].w});
                        info.T = {InterpolatedP_x/InterpolatedW, InterpolatedP_y/InterpolatedW};

                        FrameBuffer.pixel(i, j) = shade(info, float3(1.0f));
                        FrameBuffer.depth(i, j) = depth;
                        FrameBuffer.valid(i, j);
                    }
                }
            }
        }
	}

	// ============================================================================================================
	// Helpers for raytraceTriangle
    float get3x3Det(float3 A, float3 B, float3 C) const {
        float V = dot(A, cross(B, C));
        return V;
    }

    // process fetched normal map
    float3 processFetchedNormal(float3 fetchedNormal, float3 shadingNormal) const {
        float3 tangent = fetchedNormal - shadingNormal * dot(fetchedNormal,shadingNormal);
        //tangent = normalize(tangent);
        float3 bitangent = cross(shadingNormal,tangent);
        float invDet = 1.0f / get3x3Det(tangent, bitangent, shadingNormal);

        float3 invTangent, invBitangent, invShadingNormal;
        invTangent = cross(bitangent, shadingNormal) * invDet;
        invBitangent = cross(shadingNormal, tangent) * invDet;
        invShadingNormal = cross(tangent, bitangent) * invDet;

        float3 objectSpaceNormal;
        objectSpaceNormal.x = fetchedNormal.x * invTangent.x + fetchedNormal.y * invTangent.y + fetchedNormal.z * invTangent.z;
        objectSpaceNormal.y = fetchedNormal.x * invBitangent.x + fetchedNormal.y * invBitangent.y + fetchedNormal.z * invBitangent.z;
        objectSpaceNormal.z = fetchedNormal.x * invShadingNormal.x + fetchedNormal.y * invShadingNormal.y + fetchedNormal.z * invShadingNormal.z;
        //objectSpaceNormal = normalize(objectSpaceNormal);

        return objectSpaceNormal;
    }

	// ============================================================================================================

	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not
		
		//Cramer's Rule method
        float3 AminusB = tri.positions[0] - tri.positions[1];
        float3 AminusC = tri.positions[0] - tri.positions[2];
        float3 AminusO = tri.positions[0] - ray.o;

        float denom = get3x3Det(AminusB,AminusC,ray.d);
        float beta = get3x3Det(AminusO,AminusC,ray.d)/denom;
        float gamma = get3x3Det(AminusB,AminusO,ray.d)/denom;
        float alpha = 1.0f - beta - gamma;		
		float t = get3x3Det(AminusB,AminusC,AminusO)/denom;
		float3 barycentric = {alpha, beta, gamma};

		if(((t > tMin) && (tMax > t)) && ((1 > alpha) && (alpha > 0)) && ((1 > beta) && (beta > 0)) && ((1 > gamma) && (gamma > 0)) ) // There is an intersection
        {
            result.material = &materials[tri.idMaterial];
            float P_x = getInterpolatedAttribute(barycentric, {tri.positions[0].x, tri.positions[1].x, tri.positions[2].x});
            float P_y = getInterpolatedAttribute(barycentric, {tri.positions[0].y, tri.positions[1].y, tri.positions[2].y});
            float P_z = getInterpolatedAttribute(barycentric, {tri.positions[0].z, tri.positions[1].z, tri.positions[2].z});
            result.P = {P_x, P_y, P_z};

            //We need N for (shading) normal and T for texture coordinate. interpolation is needed.
            float T_x = getInterpolatedAttribute(barycentric, {tri.texcoords[0].x, tri.texcoords[1].x, tri.texcoords[2].x});
            float T_y = getInterpolatedAttribute(barycentric, {tri.texcoords[0].y, tri.texcoords[1].y, tri.texcoords[2].y});
            result.T = {T_x, T_y};

            float N_x = getInterpolatedAttribute(barycentric, {tri.normals[0].x, tri.normals[1].x, tri.normals[2].x});
            float N_y = getInterpolatedAttribute(barycentric, {tri.normals[0].y, tri.normals[1].y, tri.normals[2].y});
            float N_z = getInterpolatedAttribute(barycentric, {tri.normals[0].z, tri.normals[1].z, tri.normals[2].z});
            result.N = {N_x, N_y, N_z};

            //Load normal map
            if (result.material->isTexturedKs) {
                float3 fetchedNormal = result.material->fetchTextureKs(result.T);
                result.N = processFetchedNormal(fetchedNormal, result.N);
            }


            result.t = t;
            return true;
        }
		return false;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
                if (materials[i].name.compare(0, 5, "phong", 0, 5) == 0) {
                    materials[i].type = MAT_PHONG;
                }
                if (materials[i].name.compare(0, 5, "halfL", 0, 5) == 0) {
                    materials[i].type = MAT_HALF_LAMBERT;
                }
                if (materials[i].name.compare(0, 4, "toon", 0, 4) == 0) {
                    materials[i].type = MAT_TOON;
                }
                if (materials[i].name.compare(0, 5, "xtoon", 0, 5) == 0) {
                    materials[i].type = MAT_XTOON;
                }
                if (materials[i].name.compare(0, 7, "fresnel", 0, 7) == 0) {
                    materials[i].type = MAT_FRESNEL;
                }
                if (materials[i].name.compare(0, 3, "tf2", 0, 3) == 0) {
                    materials[i].type = MAT_TF2;
                }
                if (materials[i].name.compare(0, 6, "tf2Ind", 0, 6) == 0) {
                    materials[i].type = MAT_TF2_INDEPENDENT;
                }
                if (materials[i].name.compare(0, 6, "tf2Dep", 0, 6) == 0) {
                    materials[i].type = MAT_TF2_DEPENDENT;
                }
                //Is the material right?
                printf("Loaded %s == MAT_%d\n", materials[i].name.c_str(), materials[i].type);
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

    void loadTextureKs(const char* fname, const int i) {
        int comp;
        materials[i].textureKs = stbi_load(fname, &materials[i].textureKsWidth, &materials[i].textureKsHeight, &comp, 3);
        if (!materials[i].textureKs) {
            std::cerr << "Unable to load texture: " << fname << std::endl;
            return;
        }
    }

    void loadTextureNs(const char* fname, const int i) {
        int comp;
        materials[i].textureNs = stbi_load(fname, &materials[i].textureNsWidth, &materials[i].textureNsHeight, &comp, 1);
        if (!materials[i].textureNs) {
            std::cerr << "Unable to load texture: " << fname << std::endl;
            return;
        }
    }

    void loadTextureTF(const char* fname, const int i) {
        int comp;
        materials[i].textureTF = stbi_load(fname, &materials[i].textureTFWidth, &materials[i].textureTFHeight, &comp, 3);
        if (!materials[i].textureTF) {
            std::cerr << "Unable to load texture: " << fname << std::endl;
            return;
        }
    }

    void loadTextureAmbient(const char* fname, const int i) {
        int comp;
        materials[i].textureAmbient = stbi_load(fname, &materials[i].textureAmbientWidth, &materials[i].textureAmbientHeight, &comp, 3);
        if (!materials[i].textureAmbient) {
            std::cerr << "Unable to load texture: " << fname << std::endl;
            return;
        }
    }

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
                mtl.isTexturedKs = false;
                mtl.isTexturedNs = false;
                mtl.isTexturedAmbient = false;
                mtl.isTexturedTF = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			} else if (lineStr.compare(0, 6, "map_Ks", 0, 6) == 0) {
                lineStr.erase(0, 7);
                lineStr.erase(lineStr.size() - 1, 1);
                materials[i - 1].isTexturedKs = true;
                loadTextureKs((base_dir + lineStr).c_str(), i - 1);
            } else if (lineStr.compare(0, 6, "map_Ns", 0, 6) == 0) {
                lineStr.erase(0, 7);
                lineStr.erase(lineStr.size() - 1, 1);
                materials[i - 1].isTexturedNs = true;
                loadTextureNs((base_dir + lineStr).c_str(), i - 1);
            } else if (lineStr.compare(0, 6, "map_TF", 0, 6) == 0) {
                lineStr.erase(0, 7);
                lineStr.erase(lineStr.size() - 1, 1);
                materials[i - 1].isTexturedTF = true;
                loadTextureTF((base_dir + lineStr).c_str(), i - 1);
            } else if (lineStr.compare(0, 6, "map_Ka", 0, 6) == 0) {
                lineStr.erase(0, 7);
                lineStr.erase(lineStr.size() - 1, 1);
                materials[i - 1].isTexturedAmbient = true;
                loadTextureAmbient((base_dir + lineStr).c_str(), i - 1);
            }
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};



// BVH node (for A2 extra)
class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};


// ====== implement it in A2 extra ======
// fill in the missing parts
class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

private:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}


//#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
	// ====== exntend it in A2 extra ======
#ifndef SAHBVH
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index  = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	bestbboxL = bboxL;
	bestbboxR = bboxR;
#else
	// implelement SAH-BVH here
#endif

	if (obj_num <= 4) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	} else {
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}










// ====== implement it in A3 ======
// fill in the missing parts
class Particle {
public:
	float3 position = float3(0.0f);
	float3 velocity = float3(0.0f);
	float3 prevPosition = position;

	void reset() {
		position = float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f);
		velocity = 2.0f * float3((PCG32::rand() - 0.5f), 0.0f, (PCG32::rand() - 0.5f));
		prevPosition = position;
		position += velocity * deltaT;
	}

	void step() {
		float3 tempPrevPosition = position;
		float3 newPosition;
		// update the particle position and velocity here

		newPosition = position + (position - prevPosition) + deltaT*deltaT*globalGravity;

		//Update positions
        position = newPosition;
		prevPosition = tempPrevPosition;
	}
};


class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	const char* sphereMeshFilePath = 0;
	float sphereSize = 0.0f;
	ParticleSystem() {};

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[1] = particles[i].position;
				particlesMesh.triangles[i].positions[0] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath)) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}
		updateMesh();
	}

	void step() {
		// add some particle-particle interaction here
		// spherical particles can be implemented here
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].step();
		}
		updateMesh();
	}
};
static ParticleSystem globalParticleSystem;








// scene definition
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<BVH> bvhs;

	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc() {
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
		}
	}

	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	// camera -> screen matrix (given to you for A1)
	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f };

		return m;
	}

	// model -> camera matrix (given to you for A1)
	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// ====== implement it in A1 ======
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		const float4x4 plm = mul(pm, lm);

		FrameBuffer.clear();
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
				objects[n]->rasterizeTriangle(objects[n]->triangles[k], plm);
			}
		}
	}

	// eye ray generation (given to you for A2)
	Ray eyeRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}

    float3 fetchEnv(float3 d) const{
        float3 color;

        if (EnvMap.loaded == true)
        {
            float r = (1/PI) * acos(d.z) / sqrt(d.x * d.x + d.y * d.y);
            float2 tex = {d.x*r, d.y*r}; // uv is [-1, 1]
            tex = (tex + 1.0f) * 0.5f; // now it is [0, 1]
            int x = int(tex.x * EnvMap.width);
            int y = int(tex.y * EnvMap.height);
            color = EnvMap.pixel(x, y);
        }
        else {color = float3(0.0f);}

        return color;
    }

	// ray tracing
	void Raytrace() const {
		FrameBuffer.clear();

		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;
				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				} else {
					if(EnvMap.loaded == true)
					{
						//If ray hits nothing, return a value from image
						FrameBuffer.pixel(i, j) = fetchEnv(ray.d);
					}
					else
					{
						FrameBuffer.pixel(i, j) = float3(0.0f);
					}
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}

};
static Scene globalScene;



// ============================================================================================================
// Helpers for shade
static float3 reflect(const float3& incident, const float3& normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

static bool checkIsReflect(const HitInfo& hit, const float3& viewDir){
    float3 reflectedDir = reflect(viewDir, hit.N);
    Ray reflectionRay = Ray(hit.P, -reflectedDir);
    HitInfo reflectionHitInfo;
    if(globalScene.intersect(reflectionHitInfo, reflectionRay, 0.000001f)){return true;}
    else {return false;}
}

static bool checkIsIncidentExiting(const float3 incident, const float3 normal) {
    //incident vector is exiting from the surface (etaO)?
    if (dot(incident, normal) > 0.0f) {return true;} else { return false;}
}

static bool refract(const float3& incident, const float3& normal, float ior, float3& refractedDir) {
    float etaI, etaO, eta;
    float3 vecN;
    float cosThetaI, k;

    // If bug happens, check whether if eta1/eta2 is reversed (which one is the material?)
    if (checkIsIncidentExiting(incident, normal))
    {
        vecN = -normal;
        eta = ior;
    }
    else
    {
        vecN = normal;
        eta = 1.0f / ior;
    }
    cosThetaI = dot(incident, vecN);
    k = 1 - (eta * eta) * (1 - cosThetaI * cosThetaI);

    if(k < 0.0f) // Total internal reflection
    {
        return false;
    }

    refractedDir = eta * (incident - cosThetaI * vecN) - (sqrt(k) * vecN);

    return true;
}

static float3 shadeDebug(const HitInfo& hit, const float3& viewDir, const int level) {
    float3 brdf, irradiance;
    // loop over all of the point light sources
    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = globalScene.pointLightSources[i]->position - hit.P;

        // the inverse-squared falloff
        const float falloff = length2(l);

        // normalize the light direction
        l /= sqrtf(falloff);

        // get the irradiance
        irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
        //brdf = hit.material->BRDF(l, viewDir, hit.N);
        brdf = hit.material->Kd / PI * irradiance;

        if (hit.material->isTextured) {
            brdf *= hit.material->fetchTexture(hit.T);
        }
        return brdf;
    }
}

static float3 shadeLambertian(const HitInfo& hit, const float3& viewDir, const int level) {
        float3 L = float3(0.0f);
        float3 brdf, irradiance;

        // loop over all of the point light sources
        for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
            float3 l = globalScene.pointLightSources[i]->position - hit.P;

            // the inverse-squared falloff
            const float falloff = length2(l);

            // normalize the light direction
            l /= sqrtf(falloff);

            // get the irradiance
            irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
            //brdf = hit.material->BRDF(l, viewDir, hit.N);
            brdf = hit.material->Kd / PI;

            if (hit.material->isTextured) {
                brdf *= hit.material->fetchTexture(hit.T);
            }
            L += irradiance * brdf;
        }
        return L;
}

static float3 shadeLambertianShadow(const HitInfo& hit, const float3& viewDir, const int level) {
    float3 L = float3(0.0f);
    float3 brdf, irradiance;

    // loop over all of the point light sources
    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = globalScene.pointLightSources[i]->position - hit.P;

        // the inverse-squared falloff
        const float falloff = length2(l);

        // normalize the light direction
        l /= sqrtf(falloff);

        // get the irradiance
        irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
        //brdf = hit.material->BRDF(l, viewDir, hit.N);
        brdf = hit.material->Kd / PI;

        if (hit.material->isTextured) {
            brdf *= hit.material->fetchTexture(hit.T);
        }

        //shadow ray
        Ray shadowRay = Ray(hit.P,l*sqrtf(falloff));
        HitInfo shadowHitInfo;
        //Set tMin to a tiny value to avoid self-intersection
        //Light length was sqrtf(falloff) so it is within the length of the light vector
        if(globalScene.intersect(shadowHitInfo, shadowRay, 0.000001f, sqrtf(falloff)))
        {
            continue;
        }
        L += irradiance * brdf;
    }
    return L;
}


static float3 shadeMetalSimple(const HitInfo& hit, const float3& viewDir, const int level) {
    if (level >= 3) {return globalScene.fetchEnv(viewDir);}

    float3 reflectedDir = reflect(viewDir, hit.N);
    // Create a reflection ray
    // Invert the direction such that -reflectedDir
    Ray reflectionRay = Ray(hit.P, -reflectedDir);
    HitInfo reflectionHitInfo;

    if(globalScene.intersect(reflectionHitInfo, reflectionRay, 0.000001f))
    {
        // Invert the direction again such that -reflectionRay.d
        //return shade(reflectionHitInfo, -reflectionRay.d, level+1);
        return hit.material->Ks * shade(reflectionHitInfo, -reflectionRay.d, level+1);
    }
    else {return globalScene.fetchEnv(reflectionRay.d);}
}

static float3 shadeGlassSimple(const HitInfo& hit, const float3& viewDir, const int level) {
    if (level >= 3) {return float3(0.0f);}

    float3 refractedDir;
    //why eta1 is supposed to be eta of the material
    // seems hit.material->eta is eta1 instead of eta2
    if (refract(viewDir, hit.N, 1.0f/hit.material->eta, refractedDir))
    {
        // Create a refraction ray
        Ray refractionRay = Ray(hit.P, -refractedDir);
        HitInfo refractionHitInfo;

        if(globalScene.intersect(refractionHitInfo, refractionRay, 0.000001f))
        {
            //Where is Kt?
            return shade(refractionHitInfo, -refractionRay.d, level+1);
        }
        else
        { return float3(0.0f); }
    }
    else //internal reflection
    {
        float3 reflectedDir = reflect(viewDir, hit.N);
        // Create a reflection ray
        // Invert the direction such that -reflectedDir
        Ray reflectionRay = Ray(hit.P, -reflectedDir);
        HitInfo reflectionHitInfo;

        if(globalScene.intersect(reflectionHitInfo, reflectionRay, 0.000001f))
        {
            // Invert the direction again such that -reflectionRay.d
            //return shade(reflectionHitInfo, -reflectionRay.d, level+1);
            return hit.material->Ks * shade(reflectionHitInfo, -reflectionRay.d, level+1);
        }
        else {return float3(0.0f);}
    }
}

static float3 shadeHalfLambertShadow(const HitInfo& hit, const float3& viewDir, const int level) {
    float3 L = float3(0.0f);
    float3 brdf, irradiance;

    // loop over all of the point light sources
    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = globalScene.pointLightSources[i]->position - hit.P;

        // the inverse-squared falloff
        const float falloff = length2(l);

        // normalize the light direction
        l /= sqrtf(falloff);

        
		//the dot product from the Lambertian model is scaled by ½, add ½ and squared.
		//The result is that this dot product, which normally lies in the range of -1 to +1,s instead in the range of 0 to 1 and has a more pleasing falloff.
		float half_lambert = float(0.5f * std::max(0.0f, dot(hit.N, l)) + 0.5f);
		half_lambert *= half_lambert;
        half_lambert = half_lambert / (4.0 * PI * falloff);

		// get the irradiance
        irradiance = half_lambert * globalScene.pointLightSources[i]->wattage;
        //brdf = hit.material->BRDF(l, viewDir, hit.N);
        brdf = hit.material->Kd / PI;

        if (hit.material->isTextured) {
            brdf *= hit.material->fetchTexture(hit.T);
        }

        //shadow ray
        Ray shadowRay = Ray(hit.P,l*sqrtf(falloff));
        HitInfo shadowHitInfo;
        //Set tMin to a tiny value to avoid self-intersection
        //Light length was sqrtf(falloff) so it is within the length of the light vector
        if(globalScene.intersect(shadowHitInfo, shadowRay, 0.000001f, sqrtf(falloff)))
        {
            continue;
        }
        L += irradiance * brdf;
    }
    return L;
}

static float3 shadePhong(const HitInfo& hit, const float3& viewDir, const int level) {
	float3 L_diffuse = float3(0.0f);
	float3 L_specular = float3(0.0f);

    float3 irradiance;
    float3 albedo = hit.material->Kd;
	float shininess = 100.0f;

    // loop over all of the point light sources
    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
		// compute the diffuse component

		float3 l = globalScene.pointLightSources[i]->position - hit.P;
		// the inverse-squared falloff
		const float falloff = length2(l);
		// normalize the light direction
		l /= sqrtf(falloff);
		// get the irradiance
        irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
        if (hit.material->isTextured) {
            albedo *= hit.material->fetchTexture(hit.T);
        }

		L_diffuse += irradiance;

		// compute the specular component
		float3 reflectedDir = reflect(-l, hit.N);
		float specularFactor = pow(std::max(0.0f, dot(reflectedDir, viewDir)), shininess);
		L_specular += globalScene.pointLightSources[i]->wattage * specularFactor;
    }

	return (albedo * L_diffuse + hit.material->Ks * L_specular)/PI;
    //return (albedo * L_diffuse + hit.material->Ks * L_specular);
}

static float3 shadeFresnel(const HitInfo& hit, const float3& viewDir, const int level, float fresnelStrength = 0.9f) {
    float3 albedo = hit.material->Kd;
    float3 reflectionColor, refractionColor;
    float transparency = 1.0f;
    float fresnelPower = 1.0f;

    float3 refractedDir, reflectedDir;

    float facingratio = dot(viewDir, hit.N);
    float fresneleffect = 1*(1-fresnelStrength) + pow(1 - facingratio, fresnelPower)*fresnelStrength;
    reflectionColor = shadeMetalSimple(hit, viewDir, level);
    refractionColor = shadeGlassSimple(hit, viewDir, level);

    float3 color = reflectionColor * fresneleffect + refractionColor * (1-fresneleffect) * transparency * albedo;
    return color;
}

static float3 shadeToonSimple(const HitInfo& hit, const float3& viewDir, const int level) {
    float3 L = float3(0.0f);
    float intensity = 0.0f;

    // loop over all of the point light sources
    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = globalScene.pointLightSources[i]->position - hit.P;

        // the inverse-squared falloff
        const float falloff = length2(l);

        // normalize the light direction
        l /= sqrtf(falloff);

        float current_intensity = std::max(0.0f, dot(hit.N, l));
        if(current_intensity > intensity)
            intensity = current_intensity;
    }

    // Hardcoded Blue color Toon
    /*
    if (intensity > 0.95)
        L = float3(0.5, 0.5, 1.0);
    else if (intensity > 0.5)
        L = float3(0.3, 0.3, 0.6);
    else if (intensity > 0.25)
        L = float3(0.2, 0.2, 0.4);
    else
        L = float3(0.1, 0.1, 0.2);
    */

    // Using Kd
    if (intensity > 0.95)
        L = hit.material->Kd;
    else if (intensity > 0.5)
        L = hit.material->Kd * 0.7f;
    else if (intensity > 0.25)
        L = hit.material->Kd * 0.35f;
    else
        L = hit.material->Kd * 0.2f;
    return L;
}

static float getSilhouetteD(float3 n, float3 v, float r=0.0f){
    //Near-silhouette attribute mapping; note that r >= 0
    // n: hit.N, v: viewDir, r: toonExponent(magnitude)
    float silhouette = std::max(0.0f, dot(n, v));
    float D = pow(silhouette, r);
    return D;
}

static float getHighlightD(float3 r, float3 v, float s=1.0f){
    //Highlight attribute mapping (Phong highlight model); note that s >= 1
    // r: reflectedDir, v: viewDir, s: shininess
    float highlight = std::max(0.0f, dot(r, v));
    float D = pow(highlight, s);
    return D;
}

static float3 shadeXToon(const HitInfo& hit, const float3& viewDir, const int level, float toonExponent = 2.0f) {
    float3 L = float3(0.0f);
    float3 albedo, irradiance;
    float3 xToonColor, lambertianColor;
    float toonDetail = getSilhouetteD(normalize(hit.N), normalize(viewDir), toonExponent);

    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = globalScene.pointLightSources[i]->position - hit.P;
        const float falloff = length2(l);
        l /= sqrtf(falloff);
        irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
        albedo = hit.material->Kd / PI;
        lambertianColor = irradiance * albedo;

        xToonColor = lambertianColor * toonDetail;

        L += xToonColor;
    }
    return L;
}

static float3 shadeTF2ViewIndependent(const HitInfo& hit, const float3& viewDir, const int level) {
    float3 L = float3(0.0f);
    float3 albedo = float3(1.0f), irradiance, ambient;
    float3 warp = float3(1.0f);
    float ambient_light_scalar = 0.9f;

    albedo = hit.material->Kd / PI;
    if (hit.material->isTextured) {
        albedo *= hit.material->fetchTexture(hit.T);
    }

    if (hit.material->isTexturedAmbient) {
        ambient_light_scalar = linalg::clamp(ambient_light_scalar, 0.0f, 0.99f);
        ambient = hit.material->Ka*hit.material->fetchTextureAmbient1DLookUp(ambient_light_scalar);
    }

    L += ambient;

    // loop over all of the point light sources
    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = globalScene.pointLightSources[i]->position - hit.P;

        // the inverse-squared falloff
        const float falloff = length2(l);

        // normalize the light direction
        l /= sqrtf(falloff);

        float half_lambert = float(0.5f * std::max(0.0f, dot(hit.N, l)) + 0.5f);
        if (hit.material->isTexturedTF) {
            half_lambert = linalg::clamp(half_lambert, 0.0f, 0.99f);
            warp = hit.material->fetchTextureTF1DLookUp(half_lambert);
            half_lambert = 1.0f; //reset half_lambert to 1.0f if warpping function is used
        }
        irradiance = warp * half_lambert;

        //shadow ray
        Ray shadowRay = Ray(hit.P,l*sqrtf(falloff));
        HitInfo shadowHitInfo;
        //Set tMin to a tiny value to avoid self-intersection
        //Light length was sqrtf(falloff) so it is within the length of the light vector
        if(globalScene.intersect(shadowHitInfo, shadowRay, 0.000001f, sqrtf(falloff)))
        {
            continue;
        }
        L += irradiance * globalScene.pointLightSources[i]->wattage / float3(4.0 * PI * falloff);
    }
    return albedo * L;
}

static float3 shadeTF2ViewDependent(const HitInfo& hit, const float3& viewDir, const int level) {
    float3 lightIntensity;
    float phongTermSpec, phongTermRim; //Phong highlights
    float3 ambient = hit.material->Ka;
    float3 normal = hit.N;
    float ambient_light_scalar = 0.9f;

    if (hit.material->isTexturedAmbient) {
        ambient_light_scalar = linalg::clamp(ambient_light_scalar, 0.0f, 0.99f);
        ambient = hit.material->Ka*hit.material->fetchTextureAmbient1DLookUp(ambient_light_scalar);
    }

    float k_spec = 10.0f; //the specular exponent fetched from a texture map
    float k_rim = 10.0f; //a constant exponent which controls the breadth of the rim highlights
    float f_r = pow(1 - dot(normal, viewDir), 4); //Fresnel term used to mask rim highlights
    float f_s = pow(1 - dot(normal, viewDir), 3); // an artist-tuned Fresnel term for general specular highlights
    float k_r = 3.0f; // a rim mask texture used to attenuate the contribution of the rim terms
    float k_s = 0.05f; //a specular mask painted into a texture channel

    if (hit.material->isTexturedNs) {
        float fetched = hit.material->fetchTextureNs(hit.T);
        k_s *= fetched;
    }

    float3 L = float3(0.0f);

    float rimHighlightsDir = dot(normal, globalUp);
    L += rimHighlightsDir*f_r*k_r*ambient;

    for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
        float3 l = normalize(globalScene.pointLightSources[i]->position - hit.P);
        const float falloff = length2(l);
        l /= sqrtf(falloff);
        lightIntensity = globalScene.pointLightSources[i]->wattage;

        float3 reflectDir = -reflect(l, normal);
        phongTermSpec = f_s*pow(dot(viewDir, reflectDir), k_spec);
        phongTermRim = f_r*k_r*pow(dot(viewDir, reflectDir), k_rim);

        //L += lightIntensity*k_s*phongTermSpec;
        //L += lightIntensity*k_s*phongTermRim;
        L += lightIntensity*k_s*std::max(phongTermSpec, phongTermRim);
    }

    return L;
}

static float3 shadeTF2(const HitInfo& hit, const float3& viewDir, const int level) {
    return shadeTF2ViewIndependent(hit, viewDir, level) + shadeTF2ViewDependent(hit, viewDir, level);
}

// ============================================================================================================

static float3 shade(const HitInfo& hit, const float3& viewDir, const int level) {
    if (hit.material->type == MAT_LAMBERTIAN) {
		//return shadeLambertian(hit, viewDir, level);
        return shadeLambertianShadow(hit, viewDir, level);
        //return shadeHalfLambertShadow(hit, viewDir, level);
	} else if (hit.material->type == MAT_METAL) {
        return shadeMetalSimple(hit, viewDir, level);
	} else if (hit.material->type == MAT_GLASS) {
        return shadeGlassSimple(hit, viewDir, level);
    } else if (hit.material->type == MAT_PHONG) {
        return shadePhong(hit, viewDir, level);
    } else if (hit.material->type == MAT_HALF_LAMBERT) {
        return shadeHalfLambertShadow(hit, viewDir, level);
    } else if (hit.material->type == MAT_FRESNEL) {
        return shadeFresnel(hit, viewDir, level);
    } else if (hit.material->type == MAT_TOON) {
        return shadeToonSimple(hit, viewDir, level);
    } else if (hit.material->type == MAT_XTOON) {
        return shadeXToon(hit, viewDir, level);
    } else if (hit.material->type == MAT_TF2) {
        //return shadeTF2ViewIndependent(hit, viewDir, level);
        //return shadeTF2ViewDependent(hit, viewDir, level);
        return shadeTF2(hit, viewDir, level);
    } else if (hit.material->type == MAT_TF2_INDEPENDENT) {
        return shadeTF2ViewIndependent(hit, viewDir, level);
    } else if (hit.material->type == MAT_TF2_DEPENDENT) {
        return shadeTF2ViewDependent(hit, viewDir, level);
    } else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}
}







// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome to CS488/688!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};



// main window
// you probably do not need to modify this in A0 to A3.
class CS488Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for CS488Window
	OpenGLInit glInit;

	CS488Window() {}
	virtual ~CS488Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
		}
		globalScene.preCalc();

		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles) {
				globalParticleSystem.step();
			}

			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};


