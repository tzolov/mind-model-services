package io.mindmodel.services.object.detection;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.ProtoBufGraphDefinition;
import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.object.detection.domain.ObjectDetection;
import org.tensorflow.Tensor;

import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Christian Tzolov
 */
public class ObjectDetectionService2 implements AutoCloseable {

	public static List<String> FEED_NAMES = Arrays.asList("image_tensor");

	public static List<String> FETCH_NAMES = Arrays.asList(
			ObjectDetectionOutputConverter.DETECTION_SCORES, ObjectDetectionOutputConverter.DETECTION_CLASSES,
			ObjectDetectionOutputConverter.DETECTION_BOXES, ObjectDetectionOutputConverter.NUM_DETECTIONS);

	public static List<String> FETCH_NAMES_WITH_MASKS = Arrays.asList(
			ObjectDetectionOutputConverter.DETECTION_SCORES, ObjectDetectionOutputConverter.DETECTION_CLASSES,
			ObjectDetectionOutputConverter.DETECTION_BOXES, ObjectDetectionOutputConverter.DETECTION_MASKS,
			ObjectDetectionOutputConverter.NUM_DETECTIONS);

	private final ObjectDetectionInputAdapter inputAdapter;
	private final GraphRunner objectDetectionGraph;
	private final ObjectDetectionOutputConverter outputConverter;

	public ObjectDetectionService2(ObjectDetectionInputAdapter inputAdapter,
			GraphRunner objectDetectionGraphRunner, ObjectDetectionOutputConverter outputConverter) {
		this.inputAdapter = inputAdapter;
		this.objectDetectionGraph = objectDetectionGraphRunner;
		this.outputConverter = outputConverter;
	}

	public List<ObjectDetection> detect(byte[] image) {
		Map<String, Tensor<?>> normalizedImage = this.inputAdapter.apply(image);

		Map<String, Tensor<?>> detectedObjects = this.objectDetectionGraph.apply(
				Collections.singletonMap("image_tensor", normalizedImage.get(ObjectDetectionInputAdapter.NORMALIZED_IMAGE)));

		List<List<ObjectDetection>> out = this.outputConverter.apply(detectedObjects);

		return out.get(0);
	}

	@Override
	public void close() {
		this.inputAdapter.close();
		this.objectDetectionGraph.close();
		//this.outputConverter.close();
	}

	public static void main(String[] args) throws IOException {
		String modelUri = "http://dl.bintray.com/big-data/generic/ssdlite_mobilenet_v2_coco_2018_05_09_frozen_inference_graph.pb";
		String labelUri = "http://dl.bintray.com/big-data/generic/mscoco_label_map.pbtxt";

		GraphRunner runner = new GraphRunner(FEED_NAMES, FETCH_NAMES)
				.withGraphDefinition(new ProtoBufGraphDefinition(new DefaultResourceLoader().getResource(modelUri), true));

		ObjectDetectionOutputConverter outputAdapter = new ObjectDetectionOutputConverter(
				new DefaultResourceLoader().getResource(labelUri), 0.4f, FETCH_NAMES);

		//byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/object-detection.jpg");
		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/wild-animals-15.jpg");

		try (ObjectDetectionService2 objectDetectionService2 =
					 new ObjectDetectionService2(new ObjectDetectionInputAdapter(), runner, outputAdapter)) {

			List<ObjectDetection> boza = objectDetectionService2.detect(inputImage);

			System.out.println(boza);
		}
	}
}
