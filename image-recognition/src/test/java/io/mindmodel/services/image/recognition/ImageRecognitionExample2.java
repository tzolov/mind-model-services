package io.mindmodel.services.image.recognition;

import java.io.FileOutputStream;
import java.io.IOException;

import io.mindmodel.services.common.attic.GraphicsUtils;
import org.apache.commons.io.IOUtils;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionExample2 {

	public static void main(String[] args) throws IOException {

		ImageRecognitionAugmenter augmenter = new ImageRecognitionAugmenter();

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/giant_panda_in_beijing_zoo_1.jpg");

		ImageRecognition inceptions = ImageRecognition.inception(
				"https://dl.bintray.com/big-data/generic/tensorflow_inception_graph.pb",
				224, 10, true);
		System.out.println(inceptions.recognizeMax(inputImage));
		System.out.println(inceptions.recognizeTopK(inputImage));
		System.out.println(ImageRecognition.toRecognitionResponse(inceptions.recognizeTopK(inputImage)));
		IOUtils.write(augmenter.apply(inputImage, ImageRecognition.toRecognitionResponse(inceptions.recognizeTopK(inputImage))),
				new FileOutputStream("./image-recognition/target/image-augmented-inceptions.jpg"));
		inceptions.close();

		ImageRecognition mobileNetV2 = ImageRecognition.mobileNetV2(
				"https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz#mobilenet_v2_1.4_224_frozen.pb",
				224, 10, true);
		System.out.println(mobileNetV2.recognizeMax(inputImage));
		System.out.println(mobileNetV2.recognizeTopK(inputImage));
		IOUtils.write(augmenter.apply(inputImage, ImageRecognition.toRecognitionResponse(mobileNetV2.recognizeTopK(inputImage))),
				new FileOutputStream("./image-recognition/target/image-augmented-mobilnetV2.jpg"));
		mobileNetV2.close();

		ImageRecognition mobileNetV1 = ImageRecognition.mobileNetV1(
				"http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz#mobilenet_v1_1.0_224_frozen.pb",
				224, 10, true);
		System.out.println(mobileNetV1.recognizeMax(inputImage));
		System.out.println(mobileNetV1.recognizeTopK(inputImage));
		IOUtils.write(augmenter.apply(inputImage, ImageRecognition.toRecognitionResponse(mobileNetV1.recognizeTopK(inputImage))),
				new FileOutputStream("./image-recognition/target/image-augmented-mobilnetV1.jpg"));
		mobileNetV1.close();
	}
}
