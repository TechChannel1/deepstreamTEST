#include <gst/gst.h>
#include <glib.h>
#include <iostream>

#define MAX_SOURCES 2

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;

        case GST_MESSAGE_ERROR:
        {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("Error: %s\n", error->message);
            g_error_free(error);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[])
{
    GMainLoop *loop = nullptr;
    GstElement *pipeline, *streammux, *pgie, *nvvidconv, *nvosd, *sink;
    GstBus *bus;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(nullptr, FALSE);

    pipeline = gst_pipeline_new("ds-pipeline");

    // === Streammux ===
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    g_object_set(G_OBJECT(streammux),
                 "batch-size", MAX_SOURCES,
                 "width", 1920,
                 "height", 1080,
                 "live-source", 1,
                 nullptr);

    // === Primary inference ===
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    g_object_set(G_OBJECT(pgie),
                 "config-file-path", "dstest_pgie_config.txt",
                 nullptr);

    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvid-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make("nveglglessink", "egl-sink");

    if (!pipeline || !streammux || !pgie || !nvvidconv || !nvosd || !sink) {
        g_printerr("Element creation failed\n");
        return -1;
    }

    gst_bin_add_many(GST_BIN(pipeline),
                     streammux, pgie, nvvidconv, nvosd, sink, nullptr);

    if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, sink, nullptr)) {
        g_printerr("Elements could not be linked\n");
        return -1;
    }

    // =========================================================
    // 🔥 SOURCES (2× MP4 oder Kamera)
    // =========================================================

    for (int i = 0; i < MAX_SOURCES; i++) {
        GstElement *source, *decodebin;
        gchar pad_name[16];

        // ===== OPTION A: MP4 =====
        source = gst_element_factory_make("filesrc", nullptr);
        decodebin = gst_element_factory_make("decodebin", nullptr);

        if (i == 0)
            g_object_set(source, "location", "video1.mp4", nullptr);
        else
            g_object_set(source, "location", "video2.mp4", nullptr);

        gst_bin_add_many(GST_BIN(pipeline), source, decodebin, nullptr);
        gst_element_link(source, decodebin);

        // decodebin → streammux
        g_snprintf(pad_name, 15, "sink_%u", i);
        GstPad *sinkpad = gst_element_get_request_pad(streammux, pad_name);

        g_signal_connect(decodebin, "pad-added",
            G_CALLBACK(+[] (GstElement *decodebin, GstPad *pad, gpointer data) {
                GstPad *sinkpad = (GstPad *)data;
                gst_pad_link(pad, sinkpad);
            }), sinkpad);
    }

    // =========================================================

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Pipeline running...\n");

    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);

    return 0;
}
