package io.mindmodel.services.pose.estimation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.ProtoBufGraphDefinition;
import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.pose.estimation.domain.Body;
import org.apache.commons.io.IOUtils;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.core.ExpandDims;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.image.DecodeJpeg;

/**
 * @author Christian Tzolov
 */
public class PMTest {

	public static void main(String[] args) throws IOException {

		int BATCHSIZE = 2;

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/tourists.jpg");
		byte[] inputImage2 = GraphicsUtils.loadAsByteArray("classpath:/images/tourists.jpg");

		long CHANNELS = 3;
		String modelUri = "http://dl.bintray.com/big-data/generic/2018-30-05-mobilenet_thin_graph_opt.pb";
		boolean cacheModel = true;

		GraphRunner imageLoaderGraph = new GraphRunner("RAW_IMAGE", "NORMALIZED_IMAGE")
				.withGraphDefinition(tf -> {
					Placeholder<String> rawImage = tf.withName("RAW_IMAGE").placeholder(String.class);

					System.out.println("RAW_IMAGE shape:" + rawImage.output().shape());
					Operand<Float> decodedImage = tf.dtypes.cast(
							tf.image.decodeJpeg(rawImage, DecodeJpeg.channels(CHANNELS)), Float.class);

					// Expand dimensions since the model expects images to have shape: [1, H, W, 3]
					ExpandDims<Float> ni = tf.expandDims(decodedImage, tf.constant(0));
					//ExpandDims<Float> ni = tf.withName("NORMALIZED_IMAGE").expandDims(decodedImage, tf.constant(0));

					tf.withName("NORMALIZED_IMAGE").concat(Arrays.asList(ni, ni), tf.constant(0));
				});

		GraphRunner poseEstimationGraph = new GraphRunner("image", "Openpose/concat_stage7")
				.withGraphDefinition(new ProtoBufGraphDefinition(modelUri, cacheModel));

		PoseEstimationTensorflowOutputConverter outputConverter =
				new PoseEstimationTensorflowOutputConverter(Arrays.asList("Openpose/concat_stage7"));

		Map<String, Tensor<?>> result = imageLoaderGraph
				.andThen(poseEstimationGraph)
				.apply(Collections.singletonMap("RAW_IMAGE", Tensor.create(inputImage)));

		List<List<Body>> bodiesList = outputConverter.apply(result);

		PoseEstimateImageAugmenter augmenter = new PoseEstimateImageAugmenter();

		int i = 0;
		for (List<Body> bodies : bodiesList) {
			byte[] augmentedImage = augmenter.apply(inputImage, bodies);
			IOUtils.write(augmentedImage, new FileOutputStream("./pose-estimation/target/boza" + i++ + ".jpg"));
		}

	}
}
