package io.mindmodel.services.pose.estimation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.mindmodel.services.common.GraphicsUtils;
import io.mindmodel.services.common.JsonMapperFunction;
import io.mindmodel.services.common.TensorFlowService;
import io.mindmodel.services.pose.estimation.domain.Body;
import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.util.StreamUtils;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

/**
 * @author Christian Tzolov
 */
public class PoseEstimationServiceTest {

	private static ResourceLoader resourceLoader = new DefaultResourceLoader();
	private static Resource mobilnetModel = resourceLoader.getResource("https://dl.bintray.com/big-data/generic/2018-30-05-mobilenet_thin_graph_opt.pb");
	private static Resource cmuModel = resourceLoader.getResource("https://dl.bintray.com/big-data/generic/2018-05-14-cmu-graph_opt.pb");

	@Ignore
	@Test
	public void testPoseDetectionCmu() throws IOException {

		PoseEstimationService poseEstimationService = new PoseEstimationService(cmuModel);
		poseEstimationService.getOutputConverter().setMinBodyPartCount(5);
		poseEstimationService.getOutputConverter().setTotalPafScoreThreshold(4.4f);

		// Detect body poses
		byte[] inputImage = resourceToByteArray("classpath:/images/tourists.jpg");
		List<Body> poses = poseEstimationService.detect(inputImage);

		assertThat(poses.size(), is(5));

		// Convert to JSON
		String posesJson = new JsonMapperFunction().apply(poses);

		//JSONArray expected = new JSONArray(resourceToString("classpath:/pose-tourists.json"));
		//JSONAssert.assertEquals(expected, new JSONArray(posesJson), false);

		// Augment the input image with detected poses
		byte[] augmentedImage = new PoseEstimateImageAugmenter().apply(inputImage, poses);

		byte[] expectedAugmentedImage = resourceToByteArray("classpath:/images/tourists_augmented.jpg");
		Assert.assertArrayEquals(expectedAugmentedImage, augmentedImage);
		//IOUtils.write(augmentedImage, new FileOutputStream("./target/out1.jpg"));
	}

	@Test
	public void testPoseDetectionCmu2() throws IOException {

		PoseEstimationService poseEstimationService = new PoseEstimationService(cmuModel);
		poseEstimationService.getOutputConverter().setMinBodyPartCount(5);
		poseEstimationService.getOutputConverter().setTotalPafScoreThreshold(4.4f);

		// Detect body poses
		byte[] inputImage = resourceToByteArray("classpath:/images/S1P-2018-boot-small.jpg");
		List<Body> poses = poseEstimationService.detect(inputImage);

		// Augment the input image with detected poses
		byte[] augmentedImage = new PoseEstimateImageAugmenter().apply(inputImage, poses);

		IOUtils.write(augmentedImage, new FileOutputStream("./target/S1P-2018-boot-small-augmented.jpg"));
	}

	@Test
	public void boxa() throws IOException {
		Resource modelResource = resourceLoader.getResource("https://dl.bintray.com/big-data/generic/2018-30-05-mobilenet_thin_graph_opt.pb");
		List<String> fetchNames = Arrays.asList("Openpose/concat_stage7");
		PoseEstimationTensorflowInputConverter inputConverter = new PoseEstimationTensorflowInputConverter();
		PoseEstimationTensorflowOutputConverter outputConverter = new PoseEstimationTensorflowOutputConverter(fetchNames);

		TensorFlowService tfService = new TensorFlowService(modelResource, fetchNames, true);

		Function<byte[][], List<List<Body>>> poseEstimationFunction = inputConverter.andThen(tfService).andThen(outputConverter);

		byte[][] images = new byte[][] { GraphicsUtils.toImageToBytes("classpath:/images/VikiMaxiAdi2.jpg") };

		List<List<Body>> result = poseEstimationFunction.apply(images);
		Assert.assertEquals(1, result.size());
	}

	public static String resourceToString(String resourcePath) throws IOException {
		Resource expectedPoseResponse = resourceLoader.getResource(resourcePath);
		try (InputStream is = expectedPoseResponse.getInputStream()) {
			return StreamUtils.copyToString(is, Charset.forName("UTF-8"));
		}
	}

	public static byte[] resourceToByteArray(String resourcePath) throws IOException {
		Resource expectedPoseResponse = resourceLoader.getResource(resourcePath);
		try (InputStream is = expectedPoseResponse.getInputStream()) {
			return StreamUtils.copyToByteArray(is);
		}
	}

	public static List<Body> fromJson(String jsonBodiesUri) {
		try {
			InputStream is = resourceLoader.getResource(jsonBodiesUri).getInputStream();
			return new ObjectMapper().readValue(is, new TypeReference<List<Body>>() {});
		}
		catch (IOException e) {
			throw new IllegalStateException(e);
		}
	}

}
