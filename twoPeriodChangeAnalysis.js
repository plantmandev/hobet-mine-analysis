// ANCHOR: Two Period Change Analysis
//=============================================================================================================================//
// Description: This script performs a two-period change analysis relying on unsupervised classification using Landsat imagery.
// This script was developed for use with Google Earth Engine.
//
// Author: Gabriel Guzman Blanco
// Date: December 2023
// Version: 1.1
//
// SETUP REQUIRED:
//   Before running, draw a polygon geometry in the GEE map panel and name it "geometry2".
//   This defines the mine reclamation study area boundary (Hobet Mine, Lincoln/Boone Co., WV).
//   Approximate bounds: lon -82.1 to -81.6, lat 37.85 to 38.15
//=============================================================================================================================//

// SECTION: Data Preparation //
//---------------------------//

var LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");
var LS7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2");
var LS8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var LS9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");

var naipVisParam = {
  opacity: 1,
  bands: ["R", "G", "B"],
  min: 24.86,
  max: 162.14,
  gamma: 1.105,
};
var imageVisParam = {
  opacity: 1,
  bands: ["red", "green", "blue"],
  min: 8173.8,
  max: 10708.2,
  gamma: 1,
};
var indexVisParam = {
  opacity: 1,
  bands: ["index"],
  min: -0.0585,
  max: 0.4662,
  gamma: 1,
};
var diffVisParam = {
  opacity: 1,
  bands: ["index"],
  min: -0.0544,
  max: 0.3393,
  gamma: 1,
};

//---------------Instructions--------------------//
// 1. Draw a polygon named "geometry2" in the GEE map panel for the study area
// 2. Adjust control panel settings as needed
// 3. View layers and adjust visualization parameters
// 4. Run the script — unsupervised classification will appear on the map
// 5. Export classification under Tasks (run the task closest to the bottom — most recent)
// 6. Import classification into the random sample script for accuracy assessment
//---------------Control Panel--------------------//

// geometry2 must be drawn as an import in the GEE Code Editor before running
var studyArea = geometry2;

// "Before" period
var beforeStart = "1985-01-01";
var beforeEnd   = "2005-12-31";
var beforeName  = "Leaf-off";

// "After" period
var afterStart = "2014-01-01";
var afterEnd   = "2020-12-31";
var afterName  = "Leaf-on";

// Additional filtering
var cloudThresh      = 25; // exclude scenes with >25% cloud cover
var filterStartMonth = 1;
var filterEndMonth   = 12;

// Normalized difference index bands
var firstBand  = "NIR";
var secondBand = "red";
var indexName  = "NDVI";

// Visualization
var vis            = imageVisParam;
var visIndex       = indexVisParam;
var visIndexChange = diffVisParam;
var zoomLevel      = 12;

// Unsupervised classification settings
var classes     = 8; // spectral classes for K-Means
var infoClasses = 4; // informational classes to reclassify into

//---------------------------------------------------------//

var studyAreaFC = ee.FeatureCollection(studyArea);
var bounds = studyAreaFC.style({ color: "red", fillColor: "00000000" });
Map.addLayer(bounds, {}, "Study Area", true);

// Cloud and shadow mask using QA_PIXEL band
// QA_PIXEL bits: 3=cloud (8), 4=cloud shadow (16), 5=snow (32)
function maskCloud(image) {
  var qa = image.select("QA_PIXEL");
  var mask = qa.bitwiseAnd(8).eq(0)    // cloud
              .and(qa.bitwiseAnd(16).eq(0))  // cloud shadow
              .and(qa.bitwiseAnd(32).eq(0)); // snow
  return image.updateMask(mask);
}

// Apply cloud masking
var lesscloudy5 = LS5.map(maskCloud);
var lesscloudy7 = LS7.map(maskCloud);
var lesscloudy8 = LS8.map(maskCloud);
var lesscloudy9 = LS9.map(maskCloud);

// Filter by cloud cover threshold
var lowcloud5 = lesscloudy5.filter(ee.Filter.lt("CLOUD_COVER", cloudThresh));
var lowcloud7 = lesscloudy7.filter(ee.Filter.lt("CLOUD_COVER", cloudThresh));
var lowcloud8 = lesscloudy8.filter(ee.Filter.lt("CLOUD_COVER", cloudThresh));
var lowcloud9 = lesscloudy9.filter(ee.Filter.lt("CLOUD_COVER", cloudThresh));

// Filter by spatial bounds and month range
var spatial5 = lowcloud5.filterBounds(studyArea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));
var spatial7 = lowcloud7.filterBounds(studyArea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));
var spatial8 = lowcloud8.filterBounds(studyArea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));
var spatial9 = lowcloud9.filterBounds(studyArea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));

// Standardize all collections to a common band naming scheme (LS5 convention)
var L5coll = spatial5
  .select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
  .map(function(image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

var L7coll = spatial7
  .select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
  .map(function(image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

// LS8/9 have an extra coastal aerosol band — shift indices accordingly
var L8coll = spatial8
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
  .map(function(image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

var L9coll = spatial9
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
  .map(function(image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

// Merge and split by time period
var allCollections = L5coll.merge(L7coll).merge(L8coll).merge(L9coll);
var before = allCollections.filterDate(beforeStart, beforeEnd);
var after  = allCollections.filterDate(afterStart, afterEnd);

print(before, "Before images");
print(after,  "After images");

// Compute NDVI per image then take median composite
function addNDVI(image) {
  return image.normalizedDifference([firstBand, secondBand]);
}

var beforeNDVI       = before.map(addNDVI);
var afterNDVI        = after.map(addNDVI);
var beforeMedian     = before.median();
var afterMedian      = after.median();
var beforeNDVIMedian = beforeNDVI.median().rename("index");
var afterNDVIMedian  = afterNDVI.median().rename("index");
var ndviDiff         = afterNDVIMedian.subtract(beforeNDVIMedian).rename("index");

// Mask pixels where either period has no valid data
var dataMask = beforeMedian.select("blue").gt(0)
               .and(afterMedian.select("blue").gt(0));

var exportImage = ee.Image([
  beforeNDVIMedian.rename("bNDVI"),
  afterNDVIMedian.rename("aNDVI"),
  ndviDiff.rename("dNDVI"),
]).updateMask(dataMask);

var input = ee.Image([beforeMedian, afterMedian, exportImage]).float();

print(input, "Median of before and after bands");

// ── Map setup ────────────────────────────────────────────────────────────────

Map.centerObject(studyArea, zoomLevel);

var infoClassStr = ee.List.sequence(100, 99 + infoClasses).map(function(i) {
  return ee.String(i).slice(0, 3);
});

var middle = ui.Map();
var left   = ui.root.widgets().get(0);
ui.root.clear();
ui.root.add(left);
ui.root.add(middle);
ui.Map.Linker([left, middle], "change-bounds");

// ── Classification ────────────────────────────────────────────────────────────

var training  = input.sample({ region: studyArea, scale: 30, numPixels: 5000, tileScale: 16 });
var clusterer = ee.Clusterer.wekaKMeans(classes).train(training);
var result       = input.cluster(clusterer).clip(studyArea);
var resultFilter = result.focal_median();

// ── Layers ────────────────────────────────────────────────────────────────────

var styling = { color: "00FFFF", fillColor: "00000000" };

middle.centerObject(studyArea);

left.addLayer(beforeMedian.clip(studyArea),     vis,            "Landsat bands, " + beforeName + " (median)", false);
left.addLayer(afterMedian.clip(studyArea),      vis,            "Landsat bands, " + afterName  + " (median)", false);
left.addLayer(beforeNDVIMedian.clip(studyArea), visIndex,       indexName + ", " + beforeName  + " (median)", false);
left.addLayer(afterNDVIMedian.clip(studyArea),  visIndex,       indexName + ", " + afterName   + " (median)", false);
left.addLayer(ndviDiff.clip(studyArea),         visIndexChange, "d" + indexName,                              false);
left.addLayer(resultFilter.randomVisualizer(),  {},             "All Classes",                                true);
left.addLayer(studyAreaFC.style(styling),       {},             "Study Area");

middle.addLayer(beforeMedian.clip(studyArea),     vis,            "Landsat bands, " + beforeName + " (median)");
middle.addLayer(afterMedian.clip(studyArea),      vis,            "Landsat bands, " + afterName  + " (median)", false);
middle.addLayer(beforeNDVIMedian.clip(studyArea), visIndex,       indexName + ", " + beforeName  + " (median)", false);
middle.addLayer(afterNDVIMedian.clip(studyArea),  visIndex,       indexName + ", " + afterName   + " (median)", false);
middle.addLayer(ndviDiff.clip(studyArea),         visIndexChange, "d" + indexName,                              false);
middle.addLayer(resultFilter.randomVisualizer(),  {},             "All Classes",                                false);
middle.addLayer(studyAreaFC.style(styling),       {},             "Study Area");

middle.setOptions("SATELLITE");

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

      left.addLayer(beforeMedian.clip(studyArea),                         vis,            "Landsat bands, " + beforeName + " (median)", false);
      left.addLayer(afterMedian.clip(studyArea),                          vis,            "Landsat bands, " + afterName  + " (median)", false);
      left.addLayer(beforeNDVIMedian.clip(studyArea),                     visIndex,       indexName + ", " + beforeName  + " (median)", false);
      left.addLayer(afterNDVIMedian.clip(studyArea),                      visIndex,       indexName + ", " + afterName   + " (median)", false);
      left.addLayer(ndviDiff.clip(studyArea),                             visIndexChange, "d" + indexName,                              false);
      left.addLayer(newReclass.randomVisualizer(),                        {},             "All Classes",                                false);
      left.addLayer(resultFilter.mask(unclassMask).randomVisualizer(),    {},             "Remaining Classes",                          true);
      left.addLayer(studyAreaFC.style(styling),                           {},             "Study Area");

      middle.addLayer(beforeMedian.clip(studyArea),                       vis,            "Landsat bands, " + beforeName + " (median)");
      middle.addLayer(afterMedian.clip(studyArea),                        vis,            "Landsat bands, " + afterName  + " (median)", false);
      middle.addLayer(beforeNDVIMedian.clip(studyArea),                   visIndex,       indexName + ", " + beforeName  + " (median)", false);
      middle.addLayer(afterNDVIMedian.clip(studyArea),                    visIndex,       indexName + ", " + afterName   + " (median)", false);
      middle.addLayer(ndviDiff.clip(studyArea),                           visIndexChange, "d" + indexName,                              false);
      middle.addLayer(newReclass.randomVisualizer(),                       {},             "All Classes",                               false);
      middle.addLayer(newReclass.mask(classMask).randomVisualizer(),       {},             "Newly Reclassified");
      middle.addLayer(studyAreaFC.style(styling),                          {},             "Study Area");

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

Export.image.toDrive({
  image: afterMedian,
  description: "LandsatMedianAfter",
  scale: 30,
  region: studyArea,
  maxPixels: 1e9,
});

Export.image.toDrive({
  image: beforeMedian,
  description: "LandsatMedianBefore",
  scale: 30,
  region: studyArea,
  maxPixels: 1e9,
});

Export.image.toDrive({
  image: exportImage,
  description: "IndexBands",
  scale: 30,
  region: studyArea,
  maxPixels: 1e9,
});

Export.table.toAsset({
  collection: studyAreaFC,
  description: "saveStudyArea",
});

Export.image.toAsset({
  image: resultFilter.select("cluster").toByte(),
  description: "unsupervised",
  scale: 30,
  region: studyArea,
  maxPixels: 1e9,
});
