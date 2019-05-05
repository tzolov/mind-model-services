package io.mindmodel.services.semantic.segmentation.attic;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import io.mindmodel.services.common.AutoCloseableSession;
import io.mindmodel.services.semantic.segmentation.attic.SemanticSegmentationUtils;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ExpandDims;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.dtypes.Cast;
import org.tensorflow.types.UInt8;

import static io.mindmodel.services.semantic.segmentation.NativeImageUtils.alphaBlending;
import static io.mindmodel.services.semantic.segmentation.NativeImageUtils.grayscaleToRgb;
import static io.mindmodel.services.semantic.segmentation.NativeImageUtils.normalizeMask;

/**
 * Blends the transparent maskImage on top of the input image
 *
 * @author Christian Tzolov
 */
public class AlphaBlendingOutputConverter extends AutoCloseableSession
		implements Function<Map<String, Tensor<?>>, byte[]> {

	/**
	 * Mask transparency. Between [0,1].
	 *  - 0 stands for opaque
	 *  - 1 is 100% transparency
	 */
	private float maskTransparency = 0.35f;

	@Override
	protected void doGraphDefinition(Ops tf) {

		// Input image [B, H, W, 3]
		Cast<Float> inputImageRgb = tf.dtypes.cast(tf.withName("input_image").placeholder(UInt8.class), Float.class);

		// [B, H, W]
		Cast<Float> maskPixels2D = tf.dtypes.cast(tf.withName("maskImage").placeholder(Long.class), Float.class);
		// [B, H, W, 1]
		ExpandDims<Float> maskPixels3D = tf.expandDims(maskPixels2D, tf.rank(maskPixels2D));
		// [B, H, W, 3]
		Operand<Float> maskRgb = grayscaleToRgb(tf, maskPixels3D);
		// Change maskImage to Back (0) background and White (255) detected object.
		Operand<Float> maskRgbNormalized = normalizeMask(tf, maskRgb, 255f);

		// Blend the transparent maskImage on top of the input image.
		Operand<Float> blended = alphaBlending(tf, maskRgbNormalized, inputImageRgb, tf.constant(0.65f));

		//Mul<Float> srcRgb = tf.math.mul(tf.math.div(srcRgbOrg, max), tf.constant(255f));
		//Mul<Float> z2 = tf.math.mul(srcRgb, dstRgb);

		// Remove the batch axis. TODO handle batch > 1
		// [H, W, 3]
		Squeeze<Float> blendedWithoutBatch = tf.squeeze(blended, Squeeze.axis(Arrays.asList(0L)));
		// Encode PNG
		tf.withName("blendedPng").image.encodePng(tf.dtypes.cast(blendedWithoutBatch, UInt8.class));
	}

	@Override
	public byte[] apply(Map<String, Tensor<?>> tensorMap) {

		Tensor<Long> mask = tensorMap.get(SemanticSegmentationUtils.OUTPUT_TENSOR_NAME).expect(Long.class);
		Tensor<UInt8> inputImage = tensorMap.get(SemanticSegmentationUtils.INPUT_TENSOR_NAME).expect(UInt8.class);

		List<Tensor<?>> result = this.getSession().runner()
				.feed("input_image", inputImage)
				.feed("maskImage", mask)
				.fetch("blendedPng")
				.run();

		return result.get(0).bytesValue();
	}

	public float getMaskTransparency() {
		return this.maskTransparency;
	}

	public void setMaskTransparency(float maskTransparency) {
		this.maskTransparency = maskTransparency;
	}
}
