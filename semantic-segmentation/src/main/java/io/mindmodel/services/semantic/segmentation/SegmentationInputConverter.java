package io.mindmodel.services.semantic.segmentation;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.GraphInputAdapter;
import org.tensorflow.Operand;
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
public class SegmentationInputConverter implements GraphInputAdapter<byte[]>, AutoCloseable {

	private static final long CHANNELS = 3;
	private static final float REQUIRED_INPUT_IMAGE_SIZE = 513f;
	public static final String INPUT_IMAGE_NAME = "input_image";
	public static final String RESIZED_IMAGE_NAME = "resized_image";

	private final GraphRunner imageNormalizerGraphRunner;

	public SegmentationInputConverter() {
		this.imageNormalizerGraphRunner = new GraphRunner(Arrays.asList(INPUT_IMAGE_NAME), Arrays.asList(RESIZED_IMAGE_NAME)) {
			@Override
			protected void doGraphDefinition(Ops tf) {
				Placeholder<String> input = tf.withName(INPUT_IMAGE_NAME).placeholder(String.class);
				ExtractJpegShape<Integer> imageShapeAndChannel = tf.image.extractJpegShape(input);
				Gather<Integer> imageShape = tf.gather(imageShapeAndChannel, tf.constant(new int[] { 0, 1 }), tf.constant(0));

				Cast<Float> maxSize = tf.dtypes.cast(tf.max(imageShape, tf.constant(0)), Float.class);
				Div<Float> scale = tf.math.div(tf.constant(REQUIRED_INPUT_IMAGE_SIZE), maxSize);
				Cast<Integer> newSize = tf.dtypes.cast(tf.math.mul(scale, tf.dtypes.cast(imageShape, Float.class)), Integer.class);

				final Operand<Float> decodedImage =
						tf.dtypes.cast(tf.image.decodeJpeg(input, DecodeJpeg.channels(CHANNELS)), Float.class);

				final Operand<Float> resizedImageFloat =
						tf.image.resizeBilinear(tf.expandDims(decodedImage, tf.constant(0)), newSize);

				tf.withName(RESIZED_IMAGE_NAME).dtypes.cast(resizedImageFloat, UInt8.class);
			}
		};
	}

	@Override
	public Map<String, Tensor<?>> apply(byte[] inputImage) {
		try (Tensor inputTensor = Tensor.create(inputImage)) {
			return this.imageNormalizerGraphRunner.apply(Collections.singletonMap(INPUT_IMAGE_NAME, inputTensor));
		}
	}

	@Override
	public void close() {
		if (this.imageNormalizerGraphRunner != null) {
			this.imageNormalizerGraphRunner.close();
		}
	}
}
