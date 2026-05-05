//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <flutter_litert_flex/flutter_litert_flex_plugin.h>
#include <object_detection/object_detection_plugin.h>

void fl_register_plugins(FlPluginRegistry* registry) {
  g_autoptr(FlPluginRegistrar) flutter_litert_flex_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "FlutterLitertFlexPlugin");
  flutter_litert_flex_plugin_register_with_registrar(flutter_litert_flex_registrar);
  g_autoptr(FlPluginRegistrar) object_detection_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "ObjectDetectionPlugin");
  object_detection_plugin_register_with_registrar(object_detection_registrar);
}
