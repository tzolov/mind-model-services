package io.mindmodel.services.semantic.segmentation;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.AutoCloseables;
import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.common.ImportedGraphRunner;
import io.mindmodel.services.common.GraphOutputAdapter;
import org.tensorflow.Tensor;

import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationService implements AutoCloseable {

	public static final String SEGMENTATION_GRAPH_INPUT_NAME = "ImageTensor:0";
	public static final String SEGMENTATION_GRAPH_OUTPUT_NAME = "SemanticPredictions:0";


	private final SegmentationInputConverter inputConverter;
	private final AlphaBlendingConverter alphaBlendingConverter;
	private final MaskImageConverter maskImageConverter;
	private final ImportedGraphRunner segmentationGraphRunner;

	public SemanticSegmentationService(SegmentationInputConverter inputConverter,
			MaskImageConverter maskImageConverter,
			AlphaBlendingConverter alphaBlendingConverter,
			ImportedGraphRunner importedGraphRunner) {
		this.inputConverter = inputConverter;
		this.maskImageConverter = maskImageConverter;
		this.alphaBlendingConverter = alphaBlendingConverter;
		this.segmentationGraphRunner = importedGraphRunner;
	}

	public static class ByteArrayOutputConverter implements GraphOutputAdapter<byte[]> {

		private final String tensorName;

		public ByteArrayOutputConverter(String tensorName) {
			this.tensorName = tensorName;
		}

		@Override
		public byte[] apply(Map<String, Tensor<?>> tensorMap) {
			return tensorMap.get(tensorName).bytesValue();
		}
	}

	public byte[] blendMask(byte[] image) {

		Map<String, Tensor<?>> normalizedImage = this.inputConverter.apply(image);

		Map<String, Tensor<?>> detectedMasks = this.segmentationGraphRunner.apply(
				Collections.singletonMap(this.segmentationGraphRunner.getFeedName(),
						normalizedImage.get(SegmentationInputConverter.RESIZED_IMAGE_NAME)));

		Map<String, Tensor<?>> maskImageTensor = this.maskImageConverter.apply(Collections.singletonMap("mask_pixels",
				detectedMasks.get(SEGMENTATION_GRAPH_OUTPUT_NAME).expect(Long.class)));

		Map<String, Tensor<?>> blenderInput = new HashMap<>();
		blenderInput.put("input_image", normalizedImage.get(SegmentationInputConverter.RESIZED_IMAGE_NAME));
		blenderInput.put("mask_image", maskImageTensor.get("mask_rgb"));

		Map<String, Tensor<?>> blendedTensors = this.alphaBlendingConverter.apply(blenderInput);

		byte[] blendedImage = new ByteArrayOutputConverter("blended_png").apply(blendedTensors);

		AutoCloseables.all(normalizedImage);
		AutoCloseables.all(detectedMasks);
		AutoCloseables.all(maskImageTensor);
		AutoCloseables.all(blenderInput);

		return blendedImage;
	}

	public byte[] maskImage(byte[] image) {

		Map<String, Tensor<?>> normalizedImage = this.inputConverter.apply(image);

		Map<String, Tensor<?>> detectedMasks = this.segmentationGraphRunner.apply(
				Collections.singletonMap(this.segmentationGraphRunner.getFeedName(),
						normalizedImage.get(SegmentationInputConverter.RESIZED_IMAGE_NAME)));

		Map<String, Tensor<?>> maskImageTensor = this.maskImageConverter.apply(Collections.singletonMap("mask_pixels",
				detectedMasks.get(SEGMENTATION_GRAPH_OUTPUT_NAME).expect(Long.class)));

		byte[] maskImage = new ByteArrayOutputConverter("mask_png").apply(maskImageTensor);

		AutoCloseables.all(normalizedImage);
		AutoCloseables.all(detectedMasks);
		AutoCloseables.all(maskImageTensor);

		return maskImage;
	}

	public long[][] maskPixels(byte[] image) {
		Map<String, Tensor<?>> normalizedImage = this.inputConverter.apply(image);

		Map<String, Tensor<?>> detectedMasks = this.segmentationGraphRunner.apply(
				Collections.singletonMap(this.segmentationGraphRunner.getFeedName(),
						normalizedImage.get(SegmentationInputConverter.RESIZED_IMAGE_NAME)));

		long[][] maskPixels = ((GraphOutputAdapter<long[][]>) tensorMap -> {
			Tensor<?> outputTensor = tensorMap.get(SEGMENTATION_GRAPH_OUTPUT_NAME);
			int width = (int) outputTensor.shape()[1];
			int height = (int) outputTensor.shape()[2];
			int batchSize = 1;
			long[][] maskPixels1 = outputTensor.copyTo(new long[batchSize][width][height])[0];
			return maskPixels1;
		}).apply(detectedMasks);


		AutoCloseables.all(normalizedImage);
		AutoCloseables.all(detectedMasks);

		return maskPixels;
	}

	@Override
	public void close() {
		this.alphaBlendingConverter.close();
		this.segmentationGraphRunner.close();
		this.inputConverter.close();
	}

	public static SemanticSegmentationService semanticSegmentationService(String modelUrl, int[][] colorMap, long[] labelFilter,
			float maskTransparency) {
		return new SemanticSegmentationService(
				new SegmentationInputConverter(),
				new MaskImageConverter(colorMap, labelFilter),
				new AlphaBlendingConverter(maskTransparency),
				new ImportedGraphRunner(new DefaultResourceLoader().getResource(modelUrl),
						Arrays.asList(SEGMENTATION_GRAPH_INPUT_NAME),
						Arrays.asList(SEGMENTATION_GRAPH_OUTPUT_NAME), true, false));
	}

	public static void main(String[] args) throws IOException {

		//String inputImageUri = "file:/Users/ctzolov/Dev/projects/mindmodel/mind-model-services/semantic-segmentation/src/test/resources/images/VikiMaxiAdi.jpg";
		String outputBlendedImagePath = "./semantic-segmentation/target/blendedImage.png";
		String outputMaskImagePath = "./semantic-segmentation/target/maskImage.png";


		try (SemanticSegmentationService segmentationService = SemanticSegmentationService.semanticSegmentationService(
				"http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz#frozen_inference_graph.pb"
				, SegmentationColorMap.CITYMAP_COLORMAP, null, 0.45f)
		) {
			byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/amsterdam-cityscape1.jpg");

			// 1. Mask pixels
			long[][] maskPixels = segmentationService.maskPixels(inputImage);

			// 2. Alpha Blending
			byte[] blended = segmentationService.blendMask(inputImage);
			ImageIO.write(ImageIO.read(new ByteArrayInputStream(blended)), "png", new File(outputBlendedImagePath));

			// 3. Mask Image
			byte[] maskImage = segmentationService.maskImage(inputImage);
			ImageIO.write(ImageIO.read(new ByteArrayInputStream(maskImage)), "png", new File(outputMaskImagePath));
		}


		try (SemanticSegmentationService segmentationService = SemanticSegmentationService.semanticSegmentationService(
				"http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz#frozen_inference_graph.pb",
				SegmentationColorMap.ADE20K_COLORMAP, null, 0.45f)
		) {
			byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/interior.jpg");

			// 1. Mask pixels
			long[][] maskPixels = segmentationService.maskPixels(inputImage);

			// 2. Alpha Blending
			byte[] blended = segmentationService.blendMask(inputImage);
			ImageIO.write(ImageIO.read(new ByteArrayInputStream(blended)), "png",
					new File("./semantic-segmentation/target/inventory-blendedImage.png"));

			// 3. Mask Image
			byte[] maskImage = segmentationService.maskImage(inputImage);
			ImageIO.write(ImageIO.read(new ByteArrayInputStream(maskImage)), "png",
					new File("./semantic-segmentation/target/inventory-MaskImage.png"));
		}

		try (SemanticSegmentationService segmentationService = SemanticSegmentationService.semanticSegmentationService(
				"http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz#frozen_inference_graph.pb",
				SegmentationColorMap.BLACK_WHITE_COLORMAP, null, 0.45f)
		) {
			byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");

			// 1. Mask pixels
			long[][] maskPixels = segmentationService.maskPixels(inputImage);

			// 2. Alpha Blending
			byte[] blended = segmentationService.blendMask(inputImage);
			ImageIO.write(ImageIO.read(new ByteArrayInputStream(blended)), "png",
					new File("./semantic-segmentation/target/pascal-blendedImage.png"));

			// 3. Mask Image
			byte[] maskImage = segmentationService.maskImage(inputImage);
			ImageIO.write(ImageIO.read(new ByteArrayInputStream(maskImage)), "png",
					new File("./semantic-segmentation/target/pascal-MaskImage.png"));
		}

	}
}
