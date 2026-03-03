/**
 * Custom bounding-box parser for ball_detector (ONNX/Engine).
 * Build inside DeepStream container; output libnvdsinfer_custom_ball_parser.so
 * into models/ and set in config_infer_primary.txt:
 *   custom-lib-path=/app/models/libnvdsinfer_custom_ball_parser.so
 *   parse-bbox-func-name=NvDsInferParseCustomBallDetector
 */
#include "nvdsinfer_custom_impl.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static bool s_debug_done = false;

static void debug_print_layers(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo)
{
  if (s_debug_done) return;
  const char *env = getenv("CUSTOM_PARSE_DEBUG");
  if (!env || (env[0] != '1' && env[0] != 'y' && env[0] != 'Y')) return;

  fprintf(stderr, "[custom_parse_ball] network: %u x %u, channels %u\n",
      networkInfo.width, networkInfo.height, networkInfo.channels);
  for (size_t i = 0; i < outputLayersInfo.size(); i++) {
    NvDsInferLayerInfo const &info = outputLayersInfo[i];
    NvDsInferDims const &d = info.inferDims;
    fprintf(stderr, "[custom_parse_ball] layer[%zu] name=%s numDims=%d numElements=%u dims=",
        i, info.layerName ? info.layerName : "(null)", d.numDims, d.numElements);
    for (unsigned k = 0; k < (unsigned)d.numDims && k < 8u; k++)
      fprintf(stderr, "%d ", d.d[k]);
    fprintf(stderr, "\n");
    if (info.buffer && d.numElements > 0) {
      float const *f = (float const *)info.buffer;
      unsigned n = (d.numElements < 20u) ? d.numElements : 20u;
      fprintf(stderr, "[custom_parse_ball] first %u floats: ", n);
      for (unsigned j = 0; j < n; j++) fprintf(stderr, "%.4f ", (double)f[j]);
      fprintf(stderr, "\n");
    }
  }
  s_debug_done = true;
}

extern "C" bool NvDsInferParseCustomBallDetector(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
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

  for (size_t layerIdx = 0; layerIdx < outputLayersInfo.size(); layerIdx++) {
    NvDsInferLayerInfo const &info = outputLayersInfo[layerIdx];
    float const *data = (float const *)info.buffer;
    NvDsInferDims const &d = info.inferDims;
    if (!data || d.numElements < 5) continue;

    unsigned int numDets = 0;
    int stride = 5;
    bool row_major = true;

    if (d.numDims >= 2) {
      int d0 = d.d[0], d1 = d.d[1], d2 = (d.numDims >= 3) ? d.d[2] : 1;
      if (d0 == 1 && d1 == 5 && d2 > 1) {
        numDets = (unsigned)d2;
        row_major = false;
        stride = (int)numDets;
      } else if (d0 == 1 && d2 == 5 && d1 > 1) {
        numDets = (unsigned)d1;
        row_major = true;
      } else if (d1 == 5 && d0 > 1) {
        numDets = (unsigned)d0;
        row_major = true;
      } else if (d0 == 5 && d1 > 1) {
        numDets = (unsigned)d1;
        row_major = false;
        stride = (int)numDets;
      }
    }
    if (numDets == 0) {
      if (d.numElements >= 5 && d.numElements % 5 == 0)
        numDets = d.numElements / 5;
      else
        continue;
    }

    for (unsigned int n = 0; n < numDets; n++) {
      float v0, v1, v2, v3, conf;
      if (row_major) {
        v0 = data[n * 5 + 0];
        v1 = data[n * 5 + 1];
        v2 = data[n * 5 + 2];
        v3 = data[n * 5 + 3];
        conf = data[n * 5 + 4];
      } else {
        v0 = data[0 * stride + n];
        v1 = data[1 * stride + n];
        v2 = data[2 * stride + n];
        v3 = data[3 * stride + n];
        conf = data[4 * stride + n];
      }
      if (conf < thresh) continue;

      bool normalized = (v0 <= 1.5f && v1 <= 1.5f && v2 <= 1.5f && v3 <= 1.5f);
      float left, top, width, height;
      if (normalized) {
        width  = v2 * netW;
        height = v3 * netH;
        left   = v0 * netW - width  * 0.5f;
        top    = v1 * netH - height * 0.5f;
      } else {
        width  = v2;
        height = v3;
        left   = v0 - width  * 0.5f;
        top    = v1 - height * 0.5f;
      }
      if (width <= 0 || height <= 0) continue;
      if (left < 0) { width += left; left = 0; }
      if (top  < 0) { height += top;  top  = 0; }
      if (left + width  > netW) width  = netW - left;
      if (top  + height > netH) height = netH - top;
      if (width <= 0 || height <= 0) continue;

      NvDsInferObjectDetectionInfo obj;
      obj.left   = left;
      obj.top    = top;
      obj.width  = width;
      obj.height = height;
      obj.classId = 0;
      obj.detectionConfidence = conf;
      objectList.push_back(obj);
    }
    if (!objectList.empty()) break;
  }
  return true;
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Winfinite-recursion"
#endif
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBallDetector);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
