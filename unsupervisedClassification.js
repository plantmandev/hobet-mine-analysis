
var imageVisParam = {
  opacity: 1,
  bands: ["SR_B4", "SR_B3", "SR_B2"],
  min: 7200.74,
  max: 12900.26,
  gamma: 1,
};
var LS8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var LS7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2");
var LS9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2");
var LS5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2");

//-------------------Control Panel-----------------------//
var path = 11; // delete leading zeros (e.g. '013' should be entered '13')
var row = 31;
var imageDate = "2020-12-31"; // enter date in the format YEAR-MONTH-DAY.  (e.g. May 21st 2018 should be '2018-05-21')
var landsatProduct = LS8; // Landsat satellite. Must be LS5, LS7, LS8, or LS9.
var studyarea = geometry; // The area you would like to classify. You can replace this with a rectangle that you draw.
var zoomlevel = 11; // higher is 'zoomier'
var visualization = imageVisParam;

// settings for unsupervised classification
var classes = 10; // add the number of spectral classes in the unsupervised classification
var infoClasses = 3; // number of informational classes
//---------------------------------------------------------//

var studyarea = ee.FeatureCollection(studyarea);

var bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"];

var filtered = ee
  .ImageCollection(landsatProduct)
  .filterMetadata("DATE_ACQUIRED", "contains", imageDate)
  .filter(ee.Filter.eq("WRS_PATH", path))
  .filter(ee.Filter.eq("WRS_ROW", row));
var filteredImg = ee.Image(filtered.first()).select(bands);

// add it to the map, and show image info in the console.
Map.centerObject(filteredImg.geometry(), zoomlevel);
print(filteredImg, "Input Image");
Map.addLayer(filteredImg, visualization, "Input Image");
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
var training = filteredImg.sample({
  region: studyarea,
  scale: 30,
  numPixels: 5000,
});

// Instantiate the clusterer and train it.
var clusterer = ee.Clusterer.wekaKMeans(classes).train(training);

// Cluster the input using the trained clusterer.
var result = filteredImg.cluster(clusterer).clip(studyarea);

var resultFilter = result.focal_median();
var oldclasses = ee.List.sequence(0, classes - 1);
var reclassified = resultFilter;

// Display the clusters with random colors.
middle.centerObject(studyarea, zoomlevel);
middle.addLayer(filteredImg, visualization, "input image");
left.addLayer(resultFilter.randomVisualizer(), {}, "All Classes", true);
middle.addLayer(reclassified.randomVisualizer(), {}, "All Classes", false);
var bounds = ee
  .FeatureCollection(studyarea)
  .style({ color: "red", fillColor: "00000000" });
left.addLayer(bounds, {}, "Study Area", true);
middle.addLayer(bounds, {}, "Study Area", true);

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
      left.addLayer(filteredImg, imageVisParam, "image", false);
      left.addLayer(
        resultFilter.mask(unclassMask).randomVisualizer(),
        {},
        "Remaining Classes",
        true
      );
      left.addLayer(newreclass.randomVisualizer(), {}, "All Classes", false);
      middle.addLayer(filteredImg, imageVisParam, "image");
      middle.addLayer(newreclass.randomVisualizer(), {}, "All Classes", false);
      middle.addLayer(
        newreclass.mask(classMask).randomVisualizer(),
        {},
        "Newly Reclassified"
      );

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

// Export the study area as an Earth Engine Asset.
var studyarea = ee.FeatureCollection(studyarea);
Export.table.toAsset({
  collection: studyarea,
  description: "saveStudyArea",
});

// //Export to Asset
// Export.image.toAsset({
//   image: resultFilter.select('cluster').toByte(),
//   description: 'unsupervised',
//   scale: 30,
//   region: studyarea,
//   maxPixels: 1e9
// });

//https://gis.stackexchange.com/questions/337777/earth-engine-how-to-access-an-image-added-to-map-from-inside-a-function
