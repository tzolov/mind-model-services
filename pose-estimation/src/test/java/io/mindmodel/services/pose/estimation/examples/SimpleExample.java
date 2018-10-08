package io.mindmodel.services.pose.estimation.examples;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import io.mindmodel.services.common.GraphicsUtils;
import io.mindmodel.services.common.JsonMapperFunction;
import io.mindmodel.services.pose.estimation.PoseEstimateImageAugmenter;
import io.mindmodel.services.pose.estimation.PoseEstimationService;
import io.mindmodel.services.pose.estimation.domain.Body;
import org.apache.commons.io.IOUtils;

/**
 * @author Christian Tzolov
 */
public class SimpleExample {

	public static void main(String[] args) throws IOException {

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/tourists.jpg");

		PoseEstimationService poseEstimationService =
				new PoseEstimationService("https://dl.bintray.com/big-data/generic/2018-05-14-cmu-graph_opt.pb", true);

		List<Body> bodies = poseEstimationService.detect(inputImage);
		System.out.println("Body List: " + bodies);

		String bodiesJson = new JsonMapperFunction().apply(bodies);
		System.out.println("Pose JSON: " + bodiesJson);

		byte[] augmentedImage = new PoseEstimateImageAugmenter().apply(inputImage, bodies);
		IOUtils.write(augmentedImage, new FileOutputStream("./pose-estimation/target/tourists-augmented.jpg"));
	}
}
