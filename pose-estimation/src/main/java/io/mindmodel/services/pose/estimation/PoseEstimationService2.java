package io.mindmodel.services.pose.estimation;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.ProtoBufGraphDefinition;
import io.mindmodel.services.pose.estimation.domain.Body;
import org.tensorflow.Operand;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.types.UInt8;

import org.springframework.util.CollectionUtils;

/**
 * @author Christian Tzolov
 */
public class PoseEstimationService2 {

	public static final String DEFAULT_POSE_ESTIMATION_MODEL = "http://dl.bintray.com/big-data/generic/2018-30-05-mobilenet_thin_graph_opt.pb";
	public static final List<String> FETCH_NAMES = Arrays.asList("Openpose/concat_stage7");

	private final Function<byte[][], List<List<Body>>> poseEstimationFunction;

	private final PoseEstimationTensorflowInputConverter inputConverter;
	private final PoseEstimationTensorflowOutputConverter outputConverter;
	private final GraphRunner tfService;

	/**
	 * By default the service will use the mobilenet_thin_graph_opt.pb model and will cache the model locally
	 */
	public PoseEstimationService2() {
		this(DEFAULT_POSE_ESTIMATION_MODEL, true);
	}

	/**
	 *
	 * @param modelUri pre-trained model URI
	 * @param cacheModel if set to true the pre-trained model is cached on the local file system
	 */
	public PoseEstimationService2(String modelUri, boolean cacheModel) {
		this(new PoseEstimationTensorflowInputConverter(), new PoseEstimationTensorflowOutputConverter(FETCH_NAMES),
				new GraphRunner("image", FETCH_NAMES)
						.withGraphDefinition(new ProtoBufGraphDefinition(modelUri, cacheModel)));
	}

	public PoseEstimationService2(PoseEstimationTensorflowInputConverter inputConverter,
			PoseEstimationTensorflowOutputConverter outputConverter, GraphRunner tfService) {

		//this.imageLoaderGraph = new GraphRunner(RAW_IMAGE, NORMALIZED_IMAGE)
		//		.withGraphDefinition(tf -> {
		//			Placeholder<String> rawImage = tf.withName(RAW_IMAGE).placeholder(String.class);
		//			Operand<UInt8> decodedImage = tf.dtypes.cast(
		//					tf.image.decodeJpeg(rawImage, DecodeJpeg.channels(CHANNELS)), UInt8.class);
		//			// Expand dimensions since the model expects images to have shape: [1, H, W, 3]
		//			tf.withName(NORMALIZED_IMAGE).expandDims(decodedImage, tf.constant(0));
		//		});

		this.inputConverter = inputConverter;
		this.outputConverter = outputConverter;
		this.tfService = tfService;

		this.poseEstimationFunction = this.inputConverter.andThen(this.tfService).andThen(this.outputConverter);
	}

	public List<Body> detect(byte[] image) {
		List<List<Body>> batches = this.poseEstimationFunction.apply(new byte[][] { image });
		if (CollectionUtils.isEmpty(batches)) {
			return Collections.emptyList();
		}
		return batches.get(0);
	}

	public List<List<Body>> detect(byte[][] images) {
		return this.poseEstimationFunction.apply(images);
	}

	public PoseEstimationTensorflowInputConverter getInputConverter() {
		return inputConverter;
	}

	public PoseEstimationTensorflowOutputConverter getOutputConverter() {
		return outputConverter;
	}

	public GraphRunner getTfService() {
		return tfService;
	}
}
