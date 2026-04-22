// ANCHOR: Unsupervised Classification
//=============================================================================================================================//
// Description: Unsupervised K-Means classification on a single Landsat scene with an interactive reclassification UI.
// This script was developed for use with Google Earth Engine.
//
// Author: Gabriel Guzman Blanco
// Date: December 2023
// Version: 1.1
//
// SETUP REQUIRED:
//   Before running, draw a polygon geometry in the GEE map panel and name it "geometry".
//   This defines the study area boundary.
//
// SATELLITE COMPATIBILITY:
//   The bands array below uses SR_B1–SR_B7, which matches LS8 and LS9.
//   For LS5 / LS7, SR_B6 does not exist as a surface reflectance band — change the
//   bands array to ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"] if switching satellites.
//=============================================================================================================================//

var LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");
var LS7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2");
var LS8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var LS9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");

var imageVisParam = {
  opacity: 1,
  bands: ["SR_B4", "SR_B3", "SR_B2"],
  min: 7200.74,
  max: 12900.26,
  gamma: 1,
};

//-------------------Control Panel-----------------------//

// geometry must be drawn as an import in the GEE Code Editor before running
var studyArea = geometry;

var path           = 19;           // WRS-2 path (no leading zeros)
var row            = 34;           // WRS-2 row
var imageDate      = "2020-12-31"; // YYYY-MM-DD
var landsatProduct = LS8;          // LS5, LS7, LS8, or LS9

// NOTE: if switching to LS5 or LS7, update the bands array below to remove SR_B6
var zoomLevel  = 11;
var visualization = imageVisParam;

// Unsupervised classification settings
var classes     = 10; // spectral classes
var infoClasses = 3;  // informational classes to reclassify into

// Band selection — valid for LS8 / LS9
// For LS5 / LS7 replace with: ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"]
var bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"];

//---------------------------------------------------------//

var studyAreaFC = ee.FeatureCollection(studyArea);

var filtered = ee.ImageCollection(landsatProduct)
  .filterMetadata("DATE_ACQUIRED", "contains", imageDate)
  .filter(ee.Filter.eq("WRS_PATH", path))
  .filter(ee.Filter.eq("WRS_ROW", row));

print(filtered.size(), "Matching scenes (expect 1)");
if (filtered.size().getInfo() === 0) {
  throw new Error("No scenes found for path " + path + " / row " + row + " / date " + imageDate);
}

// Apply QA_PIXEL cloud mask before classification
function maskCloudSingle(image) {
  var qa = image.select("QA_PIXEL");
  return image.mask(qa.bitwiseAnd(8).eq(0).and(qa.bitwiseAnd(32).eq(0)));
}
var filteredImg = ee.Image(filtered.map(maskCloudSingle).first()).select(bands);

print(filteredImg, "Input Image");
Map.centerObject(filteredImg.geometry(), zoomLevel);
Map.addLayer(filteredImg, visualization, "Input Image");

var infoClassStr = ee.List.sequence(100, 99 + infoClasses).map(function(i) {
  return ee.String(i).slice(0, 3);
});

// ── Map setup ─────────────────────────────────────────────────────────────────

var middle = ui.Map();
var left   = ui.root.widgets().get(0);
ui.root.clear();
ui.root.add(left);
ui.root.add(middle);
ui.Map.Linker([left, middle], "change-bounds");

// ── Classification ────────────────────────────────────────────────────────────

var training = filteredImg.sample({ region: studyArea, scale: 30, numPixels: 5000 });
var clusterer = ee.Clusterer.wekaKMeans(classes).train(training);
var result    = filteredImg.cluster(clusterer).clip(studyArea);
var resultFilter = result.focal_median();

// ── Layers ────────────────────────────────────────────────────────────────────

var bounds  = studyAreaFC.style({ color: "red", fillColor: "00000000" });

middle.centerObject(studyArea, zoomLevel);
middle.addLayer(filteredImg, visualization, "input image");
middle.setOptions("SATELLITE");

left.addLayer(resultFilter.randomVisualizer(), {}, "All Classes", true);
left.addLayer(bounds, {}, "Study Area", true);
middle.addLayer(resultFilter.randomVisualizer(), {}, "All Classes", false);
middle.addLayer(bounds, {}, "Study Area", true);

// ── UI panel ──────────────────────────────────────────────────────────────────

var panel = ui.Panel();
panel.style().set({ width: "230px", position: "middle-right", shown: false });
left.add(panel);

var spectralLabel = ui.Label("Spectral Classes");
spectralLabel.style().set({ width: "230px", position: "bottom-left", shown: true });
left.add(spectralLabel);

var infoLabel = ui.Label("Informational Classes");
infoLabel.style().set({ width: "230px", position: "bottom-left", shown: true });
middle.add(infoLabel);

// ── Interactive reclassification ──────────────────────────────────────────────

left.onClick(function(coords) {
  var layerNames = middle.layers().getJsArray().map(function(layer) {
    return layer.get("name");
  });
  var idx     = layerNames.indexOf("All Classes");
  var myImage = middle.layers().getJsArray()[idx].getEeObject().select("cluster");

  panel.clear();
  panel.style().set("shown", true);

  var point = ee.FeatureCollection(
    ee.Feature(ee.Geometry.Point(coords.lon, coords.lat), { label: "lat/long" })
  );
  var value = myImage.reduceRegion(ee.Reducer.first(), point, 30).get("cluster");

  middle.addLayer(point, { color: "red" }, "point");
  left.addLayer(point,   { color: "red" }, "point");

  var select = ui.Select({
    items: infoClassStr.getInfo(),
    onChange: function(rc) {
      var newClass  = myImage.remap([value], [ee.Number.parse(rc)]).rename("cluster");
      print("Old Value: " + value.getInfo() + ", New Value: " + ee.Number.parse(rc).getInfo());
      var newReclass  = newClass.unmask(myImage);
      var unclassMask = newReclass.lt(100);
      var classMask   = newReclass.gte(100);

      left.layers().reset();
      middle.layers().reset();

      left.addLayer(filteredImg,                                       imageVisParam, "image",               false);
      left.addLayer(resultFilter.mask(unclassMask).randomVisualizer(), {},            "Remaining Classes",   true);
      left.addLayer(newReclass.randomVisualizer(),                     {},            "All Classes",         false);

      middle.addLayer(filteredImg,                                    imageVisParam, "image");
      middle.addLayer(newReclass.randomVisualizer(),                  {},            "All Classes",        false);
      middle.addLayer(newReclass.mask(classMask).randomVisualizer(),  {},            "Newly Reclassified");

      Export.image.toAsset({
        image: newReclass.mask(classMask).toByte(),
        description: "unsupervisedImage",
        scale: 30,
        region: studyArea,
        maxPixels: 1e9,
      });
    },
  });
  select.setPlaceholder("Choose an informational class...");
  panel.add(select);
});

// ── Exports ───────────────────────────────────────────────────────────────────

Export.table.toAsset({
  collection: studyAreaFC,
  description: "saveStudyArea",
});
