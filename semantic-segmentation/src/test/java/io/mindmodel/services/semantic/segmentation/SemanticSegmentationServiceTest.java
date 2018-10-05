package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.GraphicsUtils;
import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationServiceTest {

	private static Resource modelADE20K = new DefaultResourceLoader().getResource("http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz#frozen_inference_graph.pb");
	private static Resource modelCITYSCAPE = new DefaultResourceLoader().getResource("http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz#frozen_inference_graph.pb");
	private static Resource modelPASCALVOC2012 = new DefaultResourceLoader().getResource("http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz#frozen_inference_graph.pb");

	@Test
	public void testADE20K() throws IOException {
		SemanticSegmentationService segmentationService = new SemanticSegmentationService(modelADE20K, true);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/interior.jpg");

		byte[] augmented = segmentationService.augment(inputImage);
		Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray(
				"classpath:/images/interior_augmented.jpg"), augmented);
		//IOUtils.write(augmented, new FileOutputStream("./target/out2.jpg"));

		byte[] masks = segmentationService.masksAsImage(inputImage);
		IOUtils.write(masks, new FileOutputStream("./target/masks.jpg"));

	}

	@Test
	public void testModelCITYSCAPE() throws IOException {
		SemanticSegmentationService segmentationService = new SemanticSegmentationService(modelCITYSCAPE, true);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/amsterdam-cityscape1.jpg");

		byte[] augmented = segmentationService.augment(inputImage);
		Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray(
				"classpath:/images/amsterdam-cityscape1_augmented.jpg"), augmented);
		//IOUtils.write(augmented, new FileOutputStream("./target/out2.jpg"));

		byte[] masks = segmentationService.masksAsImage(inputImage);

	}

	@Test
	public void testModelPASCALVOC2012() throws IOException {
		SemanticSegmentationService segmentationService = new SemanticSegmentationService(modelPASCALVOC2012, true);

		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");

		byte[] augmented = segmentationService.augment(inputImage);
		Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi_augmented.jpg"), augmented);
		//IOUtils.write(augmented, new FileOutputStream("./target/out2.jpg"));

		byte[] masks = segmentationService.masksAsImage(inputImage);
		writeImage(masks, "jpg", "./target/masks1.jpg");
//		Assert.assertArrayEquals(GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi_masks.png"), masks);
	}

	private void writeImage(byte[] image, String imageFormat, String outputPath) throws IOException {
		BufferedImage i1 = ImageIO.read(new ByteArrayInputStream(image));
		ImageIO.write(i1, imageFormat, new FileOutputStream(outputPath));
	}
}
