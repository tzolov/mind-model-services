package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.semantic.segmentation.attic.SemanticSegmentationUtils;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gather;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.ExtractJpegShape;
import org.tensorflow.op.math.Div;
import org.tensorflow.types.UInt8;

/**
 * @author Christian Tzolov
 */
public class Main {
	static final int BATCH_SIZE = 1;
	static final long CHANNELS = 3;
	static final float REQUIRED_INPUT_IMAGE_SIZE = 513f;


	public static byte[][][] compute2() throws IOException {
		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/giant_panda_in_beijing_zoo_1.jpg");

		Graph g = new Graph();

		Ops tf = Ops.create(g);  // Normalize image eagerly, using default session

		Placeholder<String> input = tf.withName("input_image").placeholder(String.class);

		ExtractJpegShape<Integer> imageShapeAndChannel = tf.image.extractJpegShape(input);

		Gather<Integer> imageShape = tf.withName("image_shape").gather(imageShapeAndChannel, tf.constant(new int[] { 0, 1 }), tf.constant(0));

		Cast<Float> maxSize = tf.dtypes.cast(tf.max(imageShape, tf.constant(0)), Float.class);

		Div<Float> scale = tf.withName("scale").math.div(tf.constant(REQUIRED_INPUT_IMAGE_SIZE), maxSize);

		Cast<Integer> newSize = tf.withName("new_size").dtypes.cast(tf.math.mul(scale, tf.dtypes.cast(imageShape, Float.class)), Integer.class);

		final Operand<Float> decodedImage =
				tf.dtypes.cast(tf.image.decodeJpeg(input.asOutput(), DecodeJpeg.channels(CHANNELS)), Float.class);

		final Operand<Float> resizedImage = tf.withName("resized_image").image.resizeBilinear(
				tf.expandDims(decodedImage, tf.constant(0)), newSize);

		Cast<UInt8> resizedImage2 = tf.withName("resized_image2").dtypes.cast(resizedImage, UInt8.class);

		Session s = new Session(g);

		try (Tensor inputTensor = Tensor.create(inputImage)) {
			List<Tensor<?>> result = s.runner()
					.feed("input_image", inputTensor)
					.fetch("image_shape")
					.fetch("scale")
					.fetch("new_size")
					.fetch("resized_image")
					.fetch("resized_image2")
					.run();

			System.out.println(result.size());
			int[] imageSize = new int[2];
			result.get(0).copyTo(imageSize);
			System.out.println(Arrays.toString(imageSize));

			System.out.println(result.get(1).floatValue()); // Scale

			int[] newImageSize = new int[2];
			result.get(2).copyTo(newImageSize);
			System.out.println(Arrays.toString(newImageSize));

			System.out.println(result.get(3));
			System.out.println(result.get(4));
			UInt8[][][][] uint8 = new UInt8[1][451][513][3];
			byte[][][][] boza = new byte[1][451][513][3];

			//long[] shape = new long[] { BATCH_SIZE, 451, 513, CHANNELS };
			//byte[] data = new byte[(int) (BATCH_SIZE * 451 * 513 * CHANNELS)];
			result.get(4).copyTo(boza);
			System.out.println(Arrays.toString(boza));
			return boza[0];
		}
	}


	public static byte[][][] compute1() throws IOException {
		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/giant_panda_in_beijing_zoo_1.jpg");
		BufferedImage scaledImage = SemanticSegmentationUtils.scaledImage(inputImage);
		Tensor<UInt8> inTensor = SemanticSegmentationUtils.createInputTensor(scaledImage);

		System.out.println(inTensor);

		byte[][][][] boza = new byte[1][451][513][3];
		inTensor.copyTo(boza);

		return boza[0];

	}


	public static void main(String[] args) throws IOException {

		byte[][][] b1 = compute2();

		System.out.println("---");

		byte[][][] b2 = compute1();

		System.out.println(b1);

	}
}
