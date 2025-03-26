// ANCHOR: Two Period Change Analysis
//=============================================================================================================================//
// Description: This script performs an two-period change analysis relying on unsupervised classification using Landsat imagery.
// This script was developed for use with Google Earth Engine.

// Author: Gabriel Guzman Blanco

// Date: December 2023

// Version: 1.0
//=============================================================================================================================//

// SECTION: Data Preparation //
//---------------------------//

var LS8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var LS7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2");
var LS9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");
var LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");
var naipVisParam = {
  opacity: 1,
  bands: ["R", "G", "B"],
  min: 24.86,
  max: 162.14,
  gamma: 1.1050000000000002,
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
  min: -0.058469999730587,
  max: 0.4661656060814858,
  gamma: 1,
};
var diffVisParam = {
  opacity: 1,
  bands: ["index"],
  min: -0.054381631420190424,
  max: 0.3393293945092145,
  gamma: 1,
};
var LS8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var LS7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2");
var LS9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");
var LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");
var LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");

//---------------Instructions--------------------//
// Adjust control panel settings as needed
// View layers and adjust visualization parameters
// Run unsupervised classification
// Export classification under tasks. Make sure to run the task closest to the bottom of the list (most recent)
// Import classification into random sample script to run accuracy assessment
//---------------Control Panel--------------------//
var studyarea = geometry2; // Set boundaries for study area

// Set the "before" period (1985-present)
var beforeStart = "1985-01-01";
var beforeEnd = "2005-12-31";
var beforename = "Leaf-off"; // Change to a logical name.

// Set the "after" period (1985-present)
var afterStart = "2014-01-01";
var afterEnd = "2020-12-31";
var aftername = "Leaf-on"; // Change to a logical name.

// Additional filtering
var cloudthresh = 25; // Exclude images with >25% cloud cover
var filterStartMonth = 1;
var filterEndMonth = 12;

// Define Normalized Difference Index (NVDI)
var firstBand = "NIR";
var secondBand = "red";
var indexname = "NDVI";

// visualization and layer settings
var vis = imageVisParam; // visualization parameters for landsat imagery
var visIndex = indexVisParam; // visualization parameters for the index (e.g. NDVI, NBR, etc.)
var visIndexChange = diffVisParam; // visualization parameters for the change of the index (e.g. dNDVI, dNBR, etc.)
var zoomlevel = 12; // higher is 'zoomier'
// var NAIPyear = 2019 // show the most recent NAIP image that is no later than this year

// settings for unsupervised classification
var classes = 8; // add the number of spectral classes in the unsupervised classification
var infoClasses = 4; // number of informational classes

//---------------------------------------------------------//
var bounds = ee
  .FeatureCollection(studyarea)
  .style({ color: "red", fillColor: "00000000" });
Map.addLayer(bounds, {}, "Study Area", true);

// make a cloud, cloud shadow, and snow mask from pixel_qa band
function maskCloud(image) {
  var qa = image.select("QA_PIXEL"); // select out the fmask band
  var mask = qa.bitwiseAnd(8).eq(0).and(
    // include shadow
    //qa.bitwiseAnd(16).eq(0)).and(                               // include snow
    qa.bitwiseAnd(32).eq(0)
  ); // include clouds
  // apply the mask to the image and return it
  return image.mask(mask); //apply the mask - 0's in mask will be excluded from computation and set to opacity=0 in display
}

// Apply spatial, temporal, cloud cover filters on LS8 data
var lesscloudy9 = LS9.map(maskCloud);
var lesscloudy8 = LS8.map(maskCloud);
var lesscloudy7 = LS7.map(maskCloud);
var lesscloudy5 = LS5.map(maskCloud);

var lowcloud9 = lesscloudy9.filter(ee.Filter.lt("CLOUD_COVER", cloudthresh));
var lowcloud8 = lesscloudy8.filter(ee.Filter.lt("CLOUD_COVER", cloudthresh));
var lowcloud7 = lesscloudy7.filter(ee.Filter.lt("CLOUD_COVER", cloudthresh));
var lowcloud5 = lesscloudy5.filter(ee.Filter.lt("CLOUD_COVER", cloudthresh));

var spatial9 = lowcloud9
  .filterBounds(studyarea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));
var spatial8 = lowcloud8
  .filterBounds(studyarea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));
var spatial7 = lowcloud7
  .filterBounds(studyarea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));
var spatial5 = lowcloud5
  .filterBounds(studyarea)
  .filter(ee.Filter.calendarRange(filterStartMonth, filterEndMonth, "month"));

// Standardize the bands to Landsat 5 system, and fill in the Landsat 7 gaps
var L5coll = ee
  .ImageCollection(spatial5)
  .select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
  .map(function (image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

var L7coll = ee
  .ImageCollection(spatial7)
  .select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
  .map(function (image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

var L8coll = ee
  .ImageCollection(spatial8)
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
  .map(function (image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

var L9coll = ee
  .ImageCollection(spatial9)
  .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
  .map(function (image) {
    return image.rename(["blue", "green", "red", "NIR", "SWIR1", "SWIR2"]);
  });

// merge collections and filter by date
// merge collections
var collection_merge = ee.ImageCollection(
  L5coll.merge(L7coll.merge(L8coll.merge(L9coll)))
); // merge L5, L7 & L8
var before = collection_merge.filterDate(beforeStart, beforeEnd);
var after = collection_merge.filterDate(afterStart, afterEnd);

print(before, "Before images");
print(after, "After images");

// Calculate index #1
function addNDVI(image) {
  var ndvi = image.normalizedDifference([firstBand, secondBand]);
  return ndvi;
}
var before_ndvi = before.map(addNDVI);
var after_ndvi = after.map(addNDVI);

// Calculate median
var before_median = before.median();
var after_median = after.median();
var before_ndvi_median = before_ndvi.median().rename("index");
var after_ndvi_median = after_ndvi.median().rename("index");
var ndvi_diff = after_ndvi_median.subtract(before_ndvi_median).rename("index");
var mask1 = before_median.add(after_median).select("blue").int();

var exportImage = ee
  .Image([
    before_ndvi_median.rename("bNDVI"),
    after_ndvi_median.rename("aNDVI"),
    ndvi_diff.rename("dNDVI"),
  ])
  .updateMask(mask1);
var input = ee.Image([before_median, after_median, exportImage]).float();

print(input, "Median of before and after bands");

// //find recent NAIP image
// var filtered = ee.ImageCollection(NAIP)
//   .filterDate('2003-01-01', NAIPyear+'-12-30')
//   .filterBounds(studyarea.geometry())
//   .sort('system:time_start', false)
//   .filter(ee.Filter.listContains('system:band_names', 'B'))
//   .select(['R', 'G', 'B']);
// var recent = ee.ImageCollection(filtered.filterBounds(studyarea.geometry()).mosaic())
// var date = ee.Date(ee.Image(filtered.first()).get('system:time_start')).format("YYYY-MM-dd");
// var dateStr = ee.String(date);

// print(recent,'NAIP Imagery for study area')

// add it to the map, and show image info in the console.
Map.centerObject(studyarea, zoomlevel);

var studyarea = ee.FeatureCollection(studyarea);
var infoClassStr = ee.List.sequence(100, 99 + infoClasses).map(function (i) {
  return ee.String(i).slice(0, 3);
});

// Add two maps to the screen.
//var left = ui.Map();
//var right = ui.Map();
var middle = ui.Map();

var root_widgets = ui.root.widgets();
var left = root_widgets.get(0);

ui.root.clear();
//ui.root.add(right);
ui.root.add(left);
ui.root.add(middle);

// Link the "change-bounds" event for the maps.
// When the user drags one map, the other will be moved in sync.
ui.Map.Linker([left, middle], "change-bounds");

// Make the training dataset.
var training = input.sample({
  region: studyarea,
  scale: 30,
  numPixels: 5000,
  tileScale: 16,
});

// Instantiate the clusterer and train it.
var clusterer = ee.Clusterer.wekaKMeans(classes).train(training);

// Cluster the input using the trained clusterer.
var result = input.cluster(clusterer).clip(studyarea);

var resultFilter = result.focal_median();
var oldclasses = ee.List.sequence(0, classes - 1);
var reclassified = resultFilter;

// Display the clusters with random colors.
middle.centerObject(studyarea);
//middle.addLayer(input,visualization,'input image')
var styling = { color: "00FFFF", fillColor: "00000000" };
//left.addLayer(filtered.mosaic().clip(studyarea),naipVisParam, 'NAIP image, '+dateStr.getInfo(), false)
left.addLayer(
  before_median.clip(studyarea),
  vis,
  "Landsat bands, " + beforename + " (median)",
  false
);
left.addLayer(
  after_median.clip(studyarea),
  vis,
  "Landsat bands, " + aftername + " (median)",
  false
);
left.addLayer(
  before_ndvi_median.clip(studyarea),
  visIndex,
  indexname + ", " + beforename + " (median)",
  false
);
left.addLayer(
  after_ndvi_median.clip(studyarea),
  visIndex,
  indexname + ", " + aftername + " (median)",
  false
);
left.addLayer(
  ndvi_diff.clip(studyarea),
  visIndexChange,
  "d" + indexname,
  false
);
left.addLayer(resultFilter.randomVisualizer(), {}, "All Classes", true);
left.addLayer(studyarea.style(styling), {}, "Study Area");

//middle.addLayer(filtered.mosaic().clip(studyarea),naipVisParam, 'NAIP image, '+dateStr.getInfo(), false)
middle.addLayer(
  before_median.clip(studyarea),
  vis,
  "Landsat bands, " + beforename + " (median)"
);
middle.addLayer(
  after_median.clip(studyarea),
  vis,
  "Landsat bands, " + aftername + " (median)",
  false
);
middle.addLayer(
  before_ndvi_median.clip(studyarea),
  visIndex,
  indexname + ", " + beforename + " (median)",
  false
);
middle.addLayer(
  after_ndvi_median.clip(studyarea),
  visIndex,
  indexname + ", " + aftername + " (median)",
  false
);
middle.addLayer(
  ndvi_diff.clip(studyarea),
  visIndexChange,
  "d" + indexname,
  false
);
middle.addLayer(reclassified.randomVisualizer(), {}, "All Classes", false);
middle.addLayer(studyarea.style(styling), {}, "Study Area");

// add it to the map, and show image info in the console.
//Map.addLayer(input,visualization, 'Input Image');

// Create a panel to hold the chart.
var panel = ui.Panel();
panel.style().set({
  width: "230px",
  position: "middle-right",
  shown: false,
});
left.add(panel);

// Create a label
var label = ui.Label("Spectral Classes");
label.style().set({
  width: "230px",
  position: "bottom-left",
  shown: true,
});
left.add(label);

// Create a label
var label = ui.Label("Image");
label.style().set({
  width: "230px",
  position: "bottom-left",
  shown: true,
});
//right.add(label);

// Create a label
var label = ui.Label("Informational Classes");
label.style().set({
  width: "230px",
  position: "bottom-left",
  shown: true,
});
middle.add(label);

middle.setOptions("SATELLITE");
//Load NAIP Data
//middle.addLayer(NAIP.select(['R', 'G', 'B']).filterBounds(studyarea).mosaic(),'NAIP')

left.onClick(function (coords) {
  //right.layers().reset();
  //right.addLayer(input,imageVisParam,'image')
  // retrieve the image
  // get the image added to the screen

  var layer_names = middle
    .layers()
    .getJsArray()
    .map(function (layer) {
      return layer.get("name");
    });
  var idx = layer_names.indexOf("All Classes"); // or use imageSelect.getValue() as layer name
  var myImage = middle.layers().getJsArray()[idx].getEeObject();
  var myImage = myImage.select("cluster");
  panel.clear();
  panel.style().set("shown", true);
  var point = ee.FeatureCollection(
    ee.Feature(ee.Geometry.Point(coords.lon, coords.lat), { label: "lat/long" })
  );
  var value = myImage
    .reduceRegion(ee.Reducer.first(), point, 30)
    .get("cluster");
  middle.addLayer(point, { color: "red" }, "point");
  //right.addLayer(point, {color:'red'},'point')
  left.addLayer(point, { color: "red" }, "point");

  var select = ui.Select({
    items: infoClassStr.getInfo(),
    onChange: function (rc) {
      var newclass = myImage
        .remap([value], [ee.Number.parse(rc)])
        .rename("cluster");
      print(
        "Old Value:" +
          value.getInfo() +
          ", New Value:" +
          ee.Number.parse(rc).getInfo()
      );
      var newreclass = newclass.unmask(myImage);
      var unclassMask = newreclass.lt(100);
      var classMask = newreclass.gte(100);
      left.layers().reset();
      middle.layers().reset();
      //right.layers().reset();

      left.addLayer(
        filtered.mosaic().clip(studyarea),
        naipVisParam,
        "NAIP image, " + dateStr.getInfo(),
        false
      );
      left.addLayer(
        before_median.clip(studyarea),
        vis,
        "Landsat bands, " + beforename + " (median)",
        false
      );
      left.addLayer(
        after_median.clip(studyarea),
        vis,
        "Landsat bands, " + aftername + " (median)",
        false
      );
      left.addLayer(
        before_ndvi_median.clip(studyarea),
        visIndex,
        indexname + ", " + beforename + " (median)",
        false
      );
      left.addLayer(
        after_ndvi_median.clip(studyarea),
        visIndex,
        indexname + ", " + aftername + " (median)",
        false
      );
      left.addLayer(
        ndvi_diff.clip(studyarea),
        visIndexChange,
        "d" + indexname,
        false
      );
      left.addLayer(newreclass.randomVisualizer(), {}, "All Classes", false);
      left.addLayer(
        resultFilter.mask(unclassMask).randomVisualizer(),
        {},
        "Remaining Classes",
        true
      );
      left.addLayer(studyarea.style(styling), {}, "Study Area");

      middle.addLayer(
        filtered.mosaic().clip(studyarea),
        naipVisParam,
        "NAIP image, " + dateStr.getInfo(),
        false
      );
      middle.addLayer(
        before_median.clip(studyarea),
        vis,
        "Landsat bands, " + beforename + " (median)"
      );
      middle.addLayer(
        after_median.clip(studyarea),
        vis,
        "Landsat bands, " + aftername + " (median)",
        false
      );
      middle.addLayer(
        before_ndvi_median.clip(studyarea),
        visIndex,
        indexname + ", " + beforename + " (median)",
        false
      );
      middle.addLayer(
        after_ndvi_median.clip(studyarea),
        visIndex,
        indexname + ", " + aftername + " (median)",
        false
      );
      middle.addLayer(
        ndvi_diff.clip(studyarea),
        visIndexChange,
        "d" + indexname,
        false
      );
      middle.addLayer(newreclass.randomVisualizer(), {}, "All Classes", false);
      middle.addLayer(
        newreclass.mask(classMask).randomVisualizer(),
        {},
        "Newly Reclassified"
      );
      middle.addLayer(studyarea.style(styling), {}, "Study Area");
      // Export to Google Drive
      Export.image.toAsset({
        image: newreclass.mask(classMask).toByte(),
        description: "unsupervisedImage",
        scale: 30,
        region: studyarea,
        maxPixels: 1e9,
      });
      //right.addLayer(input,imageVisParam,'image')
    },
  });
  // Set a place holder.
  select.setPlaceholder("Choose an informational class...");
  panel.add(select);
});

// Create image and export by geometry
Export.image.toDrive({
  image: after_median,
  description: "LandsatMedianAfter",
  scale: 30,
  region: studyarea,
  maxPixels: 1e9,
});

// Create image and export by geometry
Export.image.toDrive({
  image: before_median,
  description: "LandsatMedianBefore",
  scale: 30,
  region: studyarea,
  maxPixels: 1e9,
});

Export.image.toDrive({
  image: exportImage,
  description: "IndexBands",
  scale: 30,
  region: studyarea,
  maxPixels: 1e9,
});

// Export the study area as an Earth Engine Asset.
var studyarea = ee.FeatureCollection(studyarea);
Export.table.toAsset({
  collection: studyarea,
  description: "saveStudyArea",
});

//Export to Asset
Export.image.toAsset({
  image: resultFilter.select("cluster").toByte(),
  description: "unsupervised",
  scale: 30,
  region: studyarea,
  maxPixels: 1e9,
});

// Export.image.toDrive({
//   image: filtered.mosaic(),
//   description: 'NAIP',
//   scale: 1,
//   region: studyarea,
//   maxPixels: 1e9
// });

//------------------Instructions---------------------------//
// Import an image asset into this script, and define number of classes.
// Or, cut and paste this into the end of another script
