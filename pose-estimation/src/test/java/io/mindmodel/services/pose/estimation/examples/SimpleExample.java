package io.mindmodel.services.pose.estimation.examples;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import io.mindmodel.services.common.JsonMapperFunction;
import io.mindmodel.services.pose.estimation.PoseEstimateImageAugmenter;
import io.mindmodel.services.pose.estimation.PoseEstimationService;
import io.mindmodel.services.pose.estimation.domain.Body;
import org.apache.commons.io.IOUtils;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.StreamUtils;

/**
 * @author Christian Tzolov
 */
public class SimpleExample {

	public static void main(String[] args) throws IOException {

		try (InputStream is =  new DefaultResourceLoader()
				.getResource("classpath:/images/tourists.jpg").getInputStream()) {

			byte[] inputImage = StreamUtils.copyToByteArray(is);

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
}
