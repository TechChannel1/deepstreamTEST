#include "nvdsinfer_custom_impl.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h> // CUDA FP16

static bool s_debug_done = false;

static void debug_print_layers(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo)
{
    if (s_debug_done) return;
    const char *env = getenv("CUSTOM_PARSE_DEBUG");
    if (!env || (env[0] != '1' && env[0] != 'y' && env[0] != 'Y')) return;

    fprintf(stderr, "[custom_parse_ball] network: %u x %u, channels %u\n",
            networkInfo.width, networkInfo.height, networkInfo.channels);

    for (size_t i = 0; i < outputLayersInfo.size(); i++) {
        const NvDsInferLayerInfo &info = outputLayersInfo[i];
        const NvDsInferDims &d = info.inferDims;
        fprintf(stderr, "[custom_parse_ball] layer[%zu] name=%s numDims=%d numElements=%u dtype=%d dims=",
                i, info.layerName ? info.layerName : "(null)", d.numDims, d.numElements, info.dataType);
        for (unsigned k = 0; k < (unsigned)d.numDims && k < 8u; k++)
            fprintf(stderr, "%d ", d.d[k]);
        fprintf(stderr, "\n");

        if (info.buffer && d.numElements > 0) {
            unsigned n = (d.numElements < 20u) ? d.numElements : 20u;
            fprintf(stderr, "[custom_parse_ball] first %u values: ", n);
            for (unsigned j = 0; j < n; j++) {
                float val = 0.f;
                if (info.dataType == 0) { // FP32
                    float *f = (float *)info.buffer;
                    val = f[j];
                } else if (info.dataType == 1) { // FP16
                    __half *h = (__half *)info.buffer;
                    val = __half2float(h[j]);
                }
                fprintf(stderr, "%.4f ", val);
            }
            fprintf(stderr, "\n");
        }
    }
    s_debug_done = true;
}

extern "C" bool NvDsInferParseCustomBallDetector(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo,
    const NvDsInferParseDetectionParams &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    objectList.clear();
    if (outputLayersInfo.empty() || !outputLayersInfo[0].buffer) return true;

    debug_print_layers(outputLayersInfo, networkInfo);

    float thresh = 0.5f;
    if (detectionParams.numClassesConfigured > 0 && !detectionParams.perClassPreclusterThreshold.empty())
        thresh = detectionParams.perClassPreclusterThreshold[0];

    const float netW = (float)networkInfo.width;
    const float netH = (float)networkInfo.height;

    // BatchSize ableiten: outputLayer[0].numElements / (NumDet * 5)
    unsigned batchSize = 1;
    const NvDsInferLayerInfo &info0 = outputLayersInfo[0];
    if (info0.inferDims.numElements > 5) {
        // Annahme: numElements = batchSize * numDetections * 5
        batchSize = info0.inferDims.numElements / 8400 / 5; // 8400 = max det pro Frame in deinem Model
        if (batchSize < 1) batchSize = 1;
    }

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (size_t layerIdx = 0; layerIdx < outputLayersInfo.size(); layerIdx++) {
            const NvDsInferLayerInfo &info = outputLayersInfo[layerIdx];
            const NvDsInferDims &d = info.inferDims;
            if (!info.buffer || d.numElements < 5) continue;

            unsigned int numDets = 0;
            int stride = 5;
            bool row_major = true;

            unsigned int elements_per_frame = d.numElements / batchSize;
            unsigned int batchOffset = batchIdx * elements_per_frame;

            auto get_float = [&](unsigned idx) -> float {
                unsigned realIdx = batchOffset + idx;
                if (info.dataType == 0) { // FP32
                    float *f = (float *)info.buffer;
                    return f[realIdx];
                } else if (info.dataType == 1) { // FP16
                    __half *h = (__half *)info.buffer;
                    return __half2float(h[realIdx]);
                }
                return 0.f;
            };

            // Anzahl Detektionen
            if (d.numDims >= 2) {
                int d0 = d.d[0], d1 = d.d[1], d2 = (d.numDims >= 3) ? d.d[2] : 1;
                if (d0 == 1 && d1 == 5 && d2 > 1) { numDets = (unsigned)d2; row_major=false; stride=(int)numDets; }
                else if (d0 == 1 && d2 == 5 && d1 > 1) { numDets = (unsigned)d1; row_major=true; }
                else if (d1 == 5 && d0 > 1) { numDets = (unsigned)d0; row_major=true; }
                else if (d0 == 5 && d1 > 1) { numDets = (unsigned)d1; row_major=false; stride=(int)numDets; }
            }
            if (numDets == 0) {
                if (elements_per_frame >= 5 && elements_per_frame % 5 == 0)
                    numDets = elements_per_frame / 5;
                else
                    continue;
            }

            for (unsigned int n = 0; n < numDets; n++) {
                float v0, v1, v2, v3, conf;
                if (row_major) {
                    v0 = get_float(n*5+0); v1 = get_float(n*5+1);
                    v2 = get_float(n*5+2); v3 = get_float(n*5+3);
                    conf = get_float(n*5+4);
                } else {
                    v0 = get_float(0*stride+n); v1 = get_float(1*stride+n);
                    v2 = get_float(2*stride+n); v3 = get_float(3*stride+n);
                    conf = get_float(4*stride+n);
                }

                if (conf < thresh) continue;

                bool normalized = (v0 <= 1.5f && v1 <= 1.5f && v2 <= 1.5f && v3 <= 1.5f);
                float left, top, width, height;
                if (normalized) {
                    width = v2*netW; height = v3*netH;
                    left = v0*netW - width*0.5f; top = v1*netH - height*0.5f;
                } else {
                    width = v2; height = v3;
                    left = v0 - width*0.5f; top = v1 - height*0.5f;
                }

                if (width <= 0 || height <= 0) continue;
                if (left < 0) { width += left; left = 0; }
                if (top < 0) { height += top; top = 0; }
                if (left+width > netW) width = netW-left;
                if (top+height > netH) height = netH-top;
                if (width <=0 || height <=0) continue;

                NvDsInferObjectDetectionInfo obj;
                obj.left = left; obj.top = top; obj.width = width; obj.height = height;
                obj.classId = 0; obj.detectionConfidence = conf;
                objectList.push_back(obj);
            }
        }
    }

    return true;
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBallDetector);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
