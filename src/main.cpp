#include "cs488.h"
CS488Window CS488;

 
// draw something in each frame
static void draw() {
    for (int j = 0; j < globalHeight; j++) {
        for (int i = 0; i < globalWidth; i++) {
            //FrameBuffer.pixel(i, j) = float3(0.5, 0, 0.5); // Purple
            //FrameBuffer.pixel(i, j) = float3(PCG32::rand()); // noise
            //FrameBuffer.pixel(i, j) = float3(0.5f * (cos((i + globalFrameCount) * 0.1f) + 1.0f)); // moving cosine
            FrameBuffer.pixel(i, j) = float3((i + globalFrameCount)%30, (j + globalFrameCount)%30, (i + globalFrameCount)%10);
        }
    }
}
static void A0(int argc, const char* argv[]) {
    // set the function to be called in the main loop
    CS488.process = draw;
}



// setting up lighting
static PointLightSource light;
static void setupLightSource() {
    //light.position = float3(0.5f, 4.0f, 1.0f); // use this for sponza.obj
    light.position = float3(3.0f, 3.0f, 3.0f);
    light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
    globalScene.addLight(&light);
}



// ======== you probably don't need to modify below in A1 to A3 ========
// loading .obj file from the command line arguments
static TriangleMesh mesh;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        EnvMap.load("../media/uffizi_probe.hdr"); // 1500*1500. Note that its different from FrameBuffer
        //EnvMap.load("../media/stpeters_probe.hdr");

        bool objLoadSucceed = mesh.load(argv[1]);
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
    } else {
        printf("Specify .obj file in the command line arguments. Example: CS488.exe cornellbox.obj\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    globalScene.addObject(&mesh);
}
static void A1(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RASTERIZE;
}

static void A2(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RAYTRACE;
}

static void A3(int argc, const char* argv[]) {
    globalEnableParticles = true;
    setupLightSource();
    globalRenderType = RENDER_RASTERIZE;
    if (argc > 1) globalParticleSystem.sphereMeshFilePath = argv[1];
    globalParticleSystem.initialize();
}

static void RunProject(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RAYTRACE;
}
// ======== you probably don't need to modify above in A1 to A3 ========


int main(int argc, const char* argv[]) {
    //A0(argc, argv);
    //A1(argc, argv);
    //A2(argc, argv);
    //A3(argc, argv);
    RunProject(argc, argv);

    CS488.start();
}
