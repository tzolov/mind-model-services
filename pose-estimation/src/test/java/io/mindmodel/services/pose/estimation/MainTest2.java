package io.mindmodel.services.pose.estimation;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.attic.GraphicsUtils;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.core.Concat;
import org.tensorflow.op.core.Gather;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.ExtractJpegShape;
import org.tensorflow.op.io.SerializeTensor;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Sub;

/**
 * @author Christian Tzolov
 */
public class MainTest2 {

	private static final float REQUIRED_INPUT_IMAGE_SIZE = 513f;

	public static void main(String[] args) throws IOException {


		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/tourists.jpg");
		byte[] inputImage2 = GraphicsUtils.loadAsByteArray("classpath:/images/pivotalOffice.jpeg");

		long CHANNELS = 3;

		int BATCHSIZE = 2;

		GraphRunner imageLoaderGraph = new GraphRunner("RAW_IMAGE",
				Arrays.asList("NORMALIZED_IMAGE", "pad", "shape"))

				.withGraphDefinition(tf -> {
					Placeholder<String> rawImage = tf.withName("RAW_IMAGE").placeholder(String.class);

					Operand<Float> decodedImages[] = new Operand[BATCHSIZE];
					for (int batchIndex = 0; batchIndex < BATCHSIZE; batchIndex++) {

						Gather<String> singleImage = tf.gather(rawImage, tf.constant((long) batchIndex), tf.constant(0));

						ExtractJpegShape<Integer> imageShapeAndChannel = tf.image.extractJpegShape(singleImage);
						Gather<Integer> imageShape = tf.gather(imageShapeAndChannel, tf.constant(new int[] { 0, 1 }), tf.constant(0));

						Cast<Float> maxSize = tf.dtypes.cast(tf.max(imageShape, tf.constant(0)), Float.class);
						Div<Float> scale = tf.math.div(tf.constant(REQUIRED_INPUT_IMAGE_SIZE), maxSize);
						Cast<Integer> newSize = tf.dtypes.cast(tf.math.mul(scale, tf.dtypes.cast(imageShape, Float.class)), Integer.class);

						Cast<Float> decodedImage = tf.dtypes.cast(
								tf.image.decodeJpeg(singleImage, DecodeJpeg.channels(CHANNELS)), Float.class);

						final Operand<Float> resizedImageFloat =
								tf.image.resizeBilinear(tf.expandDims(decodedImage, tf.constant(0)), newSize);


						Gather<Integer> imageWidth = tf.gather(imageShapeAndChannel, tf.constant(new int[] { 0 }), tf.constant(0));
						Sub<Integer> pad = tf.withName("pad").math.sub(tf.constant(1000), imageWidth);

						int[] bla = new int[] { 0 };
						Operand<Integer> shape = tf.withName("shape").stack(Arrays.asList(tf.constant(bla), pad));

						//tf.zeros(shape, Float.class);

						//decodedImages[batchIndex] = resizedImageFloat;
						decodedImages[batchIndex] = tf.expandDims(decodedImage, tf.constant(0));
					}

					tf.withName("NORMALIZED_IMAGE").concat(Arrays.asList(decodedImages), tf.constant(0));

					//tf.withName("OUT1").nextIteration(rawImage);
					////System.out.println(rawImage.output().shape().numDimensions());
					//tf.withName("OUT2").dtypes.cast(rawImage, String.class);

					//System.out.println("RAW_IMAGE shape:" + rawImage.output().shape());
					//Operand<Float> decodedImage = tf.dtypes.cast(
					//		tf.image.decodeJpeg(rawImage, DecodeJpeg.channels(CHANNELS)), Float.class);
					//
					//// Expand dimensions since the model expects images to have shape: [1, H, W, 3]
					//ExpandDims<Float> ni = tf.expandDims(decodedImage, tf.constant(0));
					//tf.withName("NORMALIZED_IMAGE").concat(Arrays.asList(ni, ni), tf.constant(0));


				});

		Map<String, Tensor<?>> boza = imageLoaderGraph.apply(Collections.singletonMap("RAW_IMAGE",
				Tensor.create(new byte[][] { inputImage2, inputImage2 })));


		System.out.println(boza);

		Tensor<?> padT = boza.get("pad");
		System.out.println(padT);
		System.out.println(padT.copyTo(new int[1])[0]);
	}
}
