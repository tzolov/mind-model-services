package io.mindmodel.services.image.recognition;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import io.mindmodel.services.common.TensorFlowService;
import org.tensorflow.Tensor;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.StreamUtils;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionService {

	private final Function<byte[], Map<String, Tensor<?>>> inputConverter;
	private final Function<Map<String, Tensor<?>>, Map<String, Double>> outputConverter;
	private final Function<Map<String, Tensor<?>>, Map<String, Tensor<?>>> tensorFlowService;

	/**
	 *
	 * @param modelUri location of the pre-trained model to use.
	 * @param labelsUri location of the list with pre-trained categories used by the model.
	 * @param inputNodeName name of the Model's input node to send the input image to.
	 * @param outputNodeName name of the Model's output node to retrieve the predictions from.
	 * @param responseSize Max number of predictions per recognize.
	 * @param cacheModel if true the pre-trained model is cached on the local file system.
	 */
	public ImageRecognitionService(String modelUri, String labelsUri, String inputNodeName,
			String outputNodeName, int imageHeight, int imageWidth, float mean, float scale,
			int responseSize, boolean cacheModel) {
		this(new ImageRecognitionInputConverter(inputNodeName, imageHeight, imageWidth, mean, scale),
				(responseSize == 1) ?
						new ImageRecognitionOutputConverterMax(labels(labelsUri)) :
						new ImageRecognitionOutputConverterTopK(labels(labelsUri), responseSize),
				new TensorFlowService(new DefaultResourceLoader().getResource(modelUri),
						Arrays.asList(outputNodeName), cacheModel));
	}

	public ImageRecognitionService(Function<byte[], Map<String, Tensor<?>>> inputConverter,
			Function<Map<String, Tensor<?>>, Map<String, Double>> outputConverter,
			Function<Map<String, Tensor<?>>, Map<String, Tensor<?>>> tensorFlowService) {

		this.inputConverter = inputConverter;
		this.outputConverter = outputConverter;
		this.tensorFlowService = tensorFlowService;
	}

	private static List<String> labels(String labelsUri) {
		try (InputStream is = new DefaultResourceLoader().getResource(labelsUri).getInputStream()) {
			return Arrays.asList(StreamUtils.copyToString(is, Charset.forName("UTF-8")).split("\n"));
		}
		catch (IOException e) {
			throw new RuntimeException("Failed to initialize the Vocabulary", e);
		}
	}

	/**
	 * Detects a single object from a single input image encoded as byte array
	 *
	 * @param image Input image encoded as byte array
	 * @return Returns an ordered map of recognized object names along with with related probability.
	 */
	public Map<String, Double> recognizeRaw(byte[] image) {
		return this.inputConverter.andThen(this.tensorFlowService).andThen(this.outputConverter).apply(image);
	}

	public List<RecognitionResponse> recognize(byte[] image) {
		return recognizeRaw(image).entrySet().stream()
				.map(e -> new RecognitionResponse(e.getKey(), e.getValue())).collect(Collectors.toList());
	}

	public static ImageRecognitionService inception(String inceptionModelUri,
			int normalizedImageSize, int responseSize, boolean cacheModel) {
		return new ImageRecognitionService(inceptionModelUri, "classpath:/labels/inception_labels.txt",
				"input", "output",
				normalizedImageSize, normalizedImageSize, 117f, 1f, responseSize, cacheModel);
	}

	/**
	 * Convenience for MobileNetV2 pre-trained models:
	 * https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet#pretrained-models
	 *
	 * The normalized image size is always square (e.g. H=W)
	 *
	 * @param mobilenetV2ModelUri
	 * @param normalizedImageSize
	 * @param responseSize
	 * @param cacheModel
	 * @return ImageRecognitionService instance configured with a MobileNetV2 pre-trained model.
	 */
	public static ImageRecognitionService mobilenetModeV2(String mobilenetV2ModelUri,
			int normalizedImageSize, int responseSize, boolean cacheModel) {
		return new ImageRecognitionService(mobilenetV2ModelUri, "classpath:/labels/mobilenet_labels.txt",
				"input", "MobilenetV2/Predictions/Reshape_1",
				normalizedImageSize, normalizedImageSize, 0f, 127f, responseSize, cacheModel);
	}

	/**
	 * Convenience for MobileNetV1 pre-trained models:
	 * https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md#pre-trained-models
	 *
	 * @param mobilenetV1ModelUri
	 * @param normalizedImageSize
	 * @param responseSize
	 * @param cacheModel
	 * @return
	 */
	public static ImageRecognitionService mobilenetModeV1(String mobilenetV1ModelUri,
			int normalizedImageSize, int responseSize, boolean cacheModel) {
		return new ImageRecognitionService(mobilenetV1ModelUri, "classpath:/labels/mobilenet_labels.txt",
				"input", "MobilenetV1/Predictions/Reshape_1",
				normalizedImageSize, normalizedImageSize, 0f, 127f, responseSize, cacheModel);
	}
}
