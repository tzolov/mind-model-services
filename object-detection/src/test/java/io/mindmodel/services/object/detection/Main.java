package io.mindmodel.services.object.detection;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import com.google.protobuf.TextFormat;
import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.common.ImportedGraphRunner;
import io.mindmodel.services.object.detection.domain.ObjectDetection;
import io.mindmodel.services.object.detection.protos.StringIntLabelMapOuterClass;
import org.tensorflow.Tensor;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.util.StreamUtils;
import org.springframework.util.StringUtils;


/**
 * @author Christian Tzolov
 */
public class Main {

	public static List<String> FEED_NAMES = Arrays.asList("image_tensor");

	public static List<String> FETCH_NAMES = Arrays.asList(
			ObjectDetectionOutputConverter.DETECTION_SCORES, ObjectDetectionOutputConverter.DETECTION_CLASSES,
			ObjectDetectionOutputConverter.DETECTION_BOXES, ObjectDetectionOutputConverter.NUM_DETECTIONS);

	private static String[] loadLabels(Resource labelsResource) throws Exception {
		try (InputStream is = labelsResource.getInputStream()) {
			String text = StreamUtils.copyToString(is, Charset.forName("UTF-8"));
			StringIntLabelMapOuterClass.StringIntLabelMap.Builder builder =
					StringIntLabelMapOuterClass.StringIntLabelMap.newBuilder();
			TextFormat.merge(text, builder);
			StringIntLabelMapOuterClass.StringIntLabelMap proto = builder.build();

			int maxLabelId = proto.getItemList().stream()
					.map(StringIntLabelMapOuterClass.StringIntLabelMapItem::getId)
					.max(Comparator.comparing(i -> i))
					.orElse(-1);

			String[] labelIdToNameMap = new String[maxLabelId + 1];
			for (StringIntLabelMapOuterClass.StringIntLabelMapItem item : proto.getItemList()) {
				if (!StringUtils.isEmpty(item.getDisplayName())) {
					labelIdToNameMap[item.getId()] = item.getDisplayName();
				}
				else {
					// Common practice is to set the name to a MID or Synsets Id. Synset is a set of synonyms that
					// share a common meaning: https://en.wikipedia.org/wiki/WordNet
					labelIdToNameMap[item.getId()] = item.getName();
				}
			}
			return labelIdToNameMap;
		}
	}

	public static void main(String[] args) throws IOException {


		String modelUri = "http://dl.bintray.com/big-data/generic/ssdlite_mobilenet_v2_coco_2018_05_09_frozen_inference_graph.pb";
		String labelUri = "http://dl.bintray.com/big-data/generic/mscoco_label_map.pbtxt";

		ObjectDetectionInputAdapter inputAdapter = new ObjectDetectionInputAdapter();

		ImportedGraphRunner runner = new ImportedGraphRunner(new DefaultResourceLoader().getResource(modelUri),
				FEED_NAMES, FETCH_NAMES, true, false);

		ObjectDetectionOutputConverter outputAdapter = new ObjectDetectionOutputConverter(
				new DefaultResourceLoader().getResource(labelUri), 0.4f, FETCH_NAMES);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/object-detection.jpg");
		//byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/wild-animals-15.jpg");


		Map<String, Tensor<?>> normalizedImage = inputAdapter.apply(inputImage);

		Map<String, Tensor<?>> detectedObjects = runner.apply(
				Collections.singletonMap("image_tensor", normalizedImage.get(ObjectDetectionInputAdapter.NORMALIZED_IMAGE)));

		List<List<ObjectDetection>> out = outputAdapter.apply(detectedObjects);

	}
}
