package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.attic.GraphicsUtils;
import org.apache.commons.io.IOUtils;
import org.junit.Test;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationServiceTest {

	private static String modelADE20K = "http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz#frozen_inference_graph.pb";
	private static String modelCITYSCAPE = "http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz#frozen_inference_graph.pb";
	private static String modelPASCALVOC2012 = "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz#frozen_inference_graph.pb";

	@Test
	public void testADE20K() throws IOException {
		SemanticSegmentation segmentationService = new SemanticSegmentation(modelADE20K,
						SegmentationColorMap.ADE20K_COLORMAP, null, 0.45f);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/interior.jpg");

		byte[] augmented = segmentationService.blendMask(inputImage);
		//Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray("classpath:/images/interior_augmented.jpg"), augmented);
		IOUtils.write(augmented, new FileOutputStream("./target/out2.jpg"));

		byte[] masks = segmentationService.maskImage(inputImage);
		IOUtils.write(masks, new FileOutputStream("./target/masks.jpg"));

	}

	@Test
	public void testModelCITYSCAPE() throws IOException {
		SemanticSegmentation segmentationService = new SemanticSegmentation(modelCITYSCAPE,
				SegmentationColorMap.CITYMAP_COLORMAP, null, 0.45f);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/amsterdam-cityscape1.jpg");

		byte[] augmented = segmentationService.blendMask(inputImage);
		//Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray("classpath:/images/amsterdam-cityscape1_augmented.jpg"), augmented);
		IOUtils.write(augmented, new FileOutputStream("./target/out3.jpg"));

		byte[] masks = segmentationService.maskImage(inputImage);

	}

	@Test
	public void testModelPASCALVOC2012() throws IOException {
		SemanticSegmentation segmentationService = new SemanticSegmentation(modelPASCALVOC2012,
				SegmentationColorMap.ADE20K_COLORMAP, null, 0.45f);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");

		byte[] augmented = segmentationService.blendMask(inputImage);
		//Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi_augmented.jpg"), augmented);
		IOUtils.write(augmented, new FileOutputStream("./target/out2.jpg"));

		byte[] masks = segmentationService.maskImage(inputImage);
		writeImage(masks, "jpg", "./target/masks1.jpg");
//		Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi_masks.png"), masks);
	}

	private void writeImage(byte[] image, String imageFormat, String outputPath) throws IOException {
		BufferedImage i1 = ImageIO.read(new ByteArrayInputStream(image));
		ImageIO.write(i1, imageFormat, new FileOutputStream(outputPath));
	}
}
