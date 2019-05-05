package io.mindmodel.services.semantic.segmentation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.types.UInt8;

import static io.mindmodel.services.semantic.segmentation.NativeImageUtils.alphaBlending;


/**
 *
 * Graph inputs:
 *  input_image - 4D (including batch) original image (with normalized size)
 *  mask_pixels - 3D (including batch) mask
 *
 *  color_map - 2D table with label to RGB color map.
 *  mask_transparency - 0D float value between [0, 1]. 0 stands for opaque and 1 stands for 100% transparency.
 *
 * Graph outputs:
 * 	blended_png - byte array 3D image
 *
 * @author Christian Tzolov
 */
public class AlphaBlendingConverter extends GraphRunner {

	private final Tensor<Float> maskTransparencyTensor;

	public AlphaBlendingConverter(float maskTransparency) {
		super(Arrays.asList("input_image", "mask_image", "mask_transparency"), Arrays.asList("blended_png"), false);
		this.maskTransparencyTensor = Tensor.create(maskTransparency).expect(Float.class);
	}

	@Override
	public void doClose() {
		this.maskTransparencyTensor.close();
	}

	@Override
	protected void doGraphDefinition(Ops tf) {

		// Input image [B, H, W, 3]
		Cast<Float> inputImageRgb = tf.dtypes.cast(tf.withName("input_image").placeholder(UInt8.class), Float.class);

		Placeholder<Integer> a = tf.withName("mask_image").placeholder(Integer.class);
		Cast<Float> maskRgb = tf.dtypes.cast(a, Float.class);

		Squeeze<Float> inputImageRgb2 = tf.squeeze(inputImageRgb, Squeeze.axis(Arrays.asList(0L)));

		Placeholder<Float> maskTransparency = tf.withName("mask_transparency").placeholder(Float.class);

		// Blend the transparent maskImage on top of the input image.
		Operand<Float> blended = NativeImageUtils.alphaBlending(tf, maskRgb, inputImageRgb2, maskTransparency);

		// Cut
		//Operand<Boolean> condition = tf.math.equal(a, tf.zerosLike(a));
		//Operand<Float> blended = tf.where3(condition, tf.zerosLike(maskRgb), inputImageRgb2);

		// Encode PNG
		tf.withName("blended_png").image.encodePng(tf.dtypes.cast(blended, UInt8.class));
	}

	@Override
	public Map<String, Tensor<?>> apply(Map<String, Tensor<?>> feeds) {

		Map<String, Tensor<?>> all = new HashMap<>();
		all.putAll(feeds);
		all.put("mask_transparency", maskTransparencyTensor);
		return super.apply(all);
	}
}
