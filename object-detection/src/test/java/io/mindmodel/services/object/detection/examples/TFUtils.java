package io.mindmodel.services.object.detection.examples;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;

import javax.imageio.ImageIO;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import com.google.protobuf.ByteString;
import io.mindmodel.services.common.attic.GraphicsUtils;
import org.tensorflow.example.BytesList;
import org.tensorflow.example.Example;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.FloatList;
import org.tensorflow.example.Int64List;
import org.tensorflow.hadoop.util.TFRecordWriter;

import org.springframework.util.StreamUtils;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;

/**
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
 * https://github.com/sararob/tswift-detection/blob/master/convert_to_tfrecord.py
 *
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto
 * https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
 *
 * https://github.com/tensorflow/ecosystem/blob/master/hadoop/src/main/java/org/tensorflow/hadoop/util/TFRecordWriter.java
 * https://github.com/tensorflow/ecosystem/blob/master/hadoop/src/test/java/org/tensorflow/hadoop/util/TFRecordTest.java
 *
 * https://towardsdatascience.com/build-a-taylor-swift-detector-with-the-tensorflow-object-detection-api-ml-engine-and-swift-82707f5b4a56
 *
 * http://heidloff.net/article/tensorflow-object-detection-deep-learning
 *
 * @author Christian Tzolov
 */
public class TFUtils {

	public static Example createTFExample(long height, long width, String filename, String imageFormat,
			byte[] encodedImageData, float xmin, float xmax, float ymin, float ymax, String classText, int classId)
			throws UnsupportedEncodingException {

		Features features = Features.newBuilder()
				.putFeature("image/height", int64Feature(height))
				.putFeature("image/width", int64Feature(width))
				.putFeature("image/filename", byteFeature(filename))
				.putFeature("image/source_id", byteFeature(filename))
				.putFeature("image/encoded", byteFeature(encodedImageData))
				.putFeature("image/format", byteFeature(imageFormat))
				.putFeature("image/object/bbox/xmin", floatFeature(xmin))
				.putFeature("image/object/bbox/xmax", floatFeature(xmax))
				.putFeature("image/object/bbox/ymin", floatFeature(ymin))
				.putFeature("image/object/bbox/ymax", floatFeature(ymax))
				.putFeature("image/object/class/text", byteFeature(classText))
				.putFeature("image/object/class/label", int64Feature(classId))
				.build();

		return Example.newBuilder().mergeFeatures(features).build();
	}

	public static Feature int64Feature(long value) {
		return Feature.newBuilder().mergeInt64List(Int64List.newBuilder().addValue(value).build()).build();
	}

	public static Feature byteFeature(String value) throws UnsupportedEncodingException {
		return Feature.newBuilder().mergeBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom(value, "UTF-8")).build()).build();
	}

	public static Feature byteFeature(byte[] value) {
		return Feature.newBuilder().mergeBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom(value)).build()).build();
	}

	public static Feature floatFeature(float value) {
		return Feature.newBuilder().mergeFloatList(FloatList.newBuilder().addValue(value).build()).build();
	}

	// XML Object Detection Annotations
	@JsonIgnoreProperties(ignoreUnknown = true)
	public static class Annotation {
		private String folder;
		private String filename;
		private String path;
		private Source source;
		private Size size;
		private String segmented;
		private AnnotationObject object;

		public String getFolder() {
			return folder;
		}

		public void setFolder(String folder) {
			this.folder = folder;
		}

		public String getFilename() {
			return filename;
		}

		public void setFilename(String filename) {
			this.filename = filename;
		}

		public String getPath() {
			return path;
		}

		public void setPath(String path) {
			this.path = path;
		}

		public Source getSource() {
			return source;
		}

		public void setSource(Source source) {
			this.source = source;
		}

		public Size getSize() {
			return size;
		}

		public void setSize(Size size) {
			this.size = size;
		}

		public String getSegmented() {
			return segmented;
		}

		public void setSegmented(String segmented) {
			this.segmented = segmented;
		}

		public AnnotationObject getObject() {
			return object;
		}

		public void setObject(AnnotationObject object) {
			this.object = object;
		}

		@Override
		public String toString() {
			return "Annotation{" +
					"folder='" + folder + '\'' +
					", filename='" + filename + '\'' +
					", path='" + path + '\'' +
					", source=" + source +
					", size=" + size +
					", segmented='" + segmented + '\'' +
					", object=" + object +
					'}';
		}
	}

	public static class Source {
		private String database;

		public String getDatabase() {
			return database;
		}

		public void setDatabase(String database) {
			this.database = database;
		}
	}

	public static class Size {
		private int width;
		private int height;
		private int depth;

		public int getWidth() {
			return width;
		}

		public void setWidth(int width) {
			this.width = width;
		}

		public int getHeight() {
			return height;
		}

		public void setHeight(int height) {
			this.height = height;
		}

		public int getDepth() {
			return depth;
		}

		public void setDepth(int depth) {
			this.depth = depth;
		}

		@Override
		public String toString() {
			return "Size{" +
					"width=" + width +
					", height=" + height +
					", depth=" + depth +
					'}';
		}
	}

	public static class AnnotationObject {
		private String name;
		private String pose;
		private int truncated;
		private int difficult;
		private BoundingBox bndbox;

		public String getName() {
			return name;
		}

		public void setName(String name) {
			this.name = name;
		}

		public String getPose() {
			return pose;
		}

		public void setPose(String pose) {
			this.pose = pose;
		}

		public int getTruncated() {
			return truncated;
		}

		public void setTruncated(int truncated) {
			this.truncated = truncated;
		}

		public int getDifficult() {
			return difficult;
		}

		public void setDifficult(int difficult) {
			this.difficult = difficult;
		}

		public BoundingBox getBndbox() {
			return bndbox;
		}

		public void setBndbox(BoundingBox bndbox) {
			this.bndbox = bndbox;
		}

		@Override
		public String toString() {
			return "AnnotationObject{" +
					"name='" + name + '\'' +
					", pose='" + pose + '\'' +
					", truncated=" + truncated +
					", difficult=" + difficult +
					", bndbox=" + bndbox +
					'}';
		}
	}

	public static class BoundingBox {
		private long xmin;
		private long ymin;
		private long xmax;
		private long ymax;

		public long getXmin() {
			return xmin;
		}

		public void setXmin(long xmin) {
			this.xmin = xmin;
		}

		public long getYmin() {
			return ymin;
		}

		public void setYmin(long ymin) {
			this.ymin = ymin;
		}

		public long getXmax() {
			return xmax;
		}

		public void setXmax(long xmax) {
			this.xmax = xmax;
		}

		public long getYmax() {
			return ymax;
		}

		public void setYmax(long ymax) {
			this.ymax = ymax;
		}

		@Override
		public String toString() {
			return "BoundingBox{" +
					"xmin=" + xmin +
					", ymin=" + ymin +
					", xmax=" + xmax +
					", ymax=" + ymax +
					'}';
		}
	}

	public static void main(String[] args) throws IOException {
		//String xmlString = StreamUtils.copyToString(
		//		new DefaultResourceLoader().getResource("classpath:/xml/1406825_2fb58.xml").getInputStream(),
		//		Charset.forName("UTF-8"));
		//Annotation annotation = new XmlMapper().readValue(xmlString, Annotation.class);
		//System.out.println(annotation);
		boza("/Users/ctzolov/Dev/projects/tmp/images_output/Annotations");
	}


	public static void boza(String xmlAnnotationFolder) throws IOException {


		XmlMapper xmlMapper = new XmlMapper();

		TFRecordWriter tfRecordWriter = new TFRecordWriter(new RandomAccessFile("/Users/ctzolov/Dev/projects/tmp/images_output/test.test", "rw"));

		for (File file : new File(xmlAnnotationFolder).listFiles()) {
			String xmlString = StreamUtils.copyToString(new FileInputStream(file), Charset.forName("UTF-8"));
			Annotation annotation = xmlMapper.readValue(xmlString, Annotation.class);

			BufferedImage image = ImageIO.read(new FileInputStream(annotation.getPath()));

			if (image.getWidth() > 600) {
				float ratio = 600.0f / image.getWidth();
				int newHeight = (int) (image.getHeight() * ratio);
				image = scale(image, 600, newHeight);
			} else {
				image = scale(image, image.getWidth(), image.getHeight());
			}

			byte[] imageBytes = GraphicsUtils.toRawByteArray(image);
			byte[] rgbImageBytes = bgrToRgb(imageBytes);

			Example example = createTFExample(image.getHeight(), image.getWidth(), annotation.getFilename(), "jpg", rgbImageBytes,
					annotation.getObject().getBndbox().getXmin(), annotation.getObject().getBndbox().getXmax(),
					annotation.getObject().getBndbox().getYmin(), annotation.getObject().getBndbox().getYmax(),
					annotation.getObject().getName(), annotation.getObject().getName().hashCode());

			tfRecordWriter.write(example.toByteArray());
		}

	}

	public static BufferedImage scale(BufferedImage originalImage, int newWidth, int newHeight) {
		Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT);
		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_3BYTE_BGR);

		Graphics2D g2d = resizedImage.createGraphics();
		g2d.drawImage(tmpImage, 0, 0, null);
		g2d.dispose();

		return resizedImage;
	}

	public static byte[] bgrToRgb(byte[] brgImage) {
		byte[] rgbImage = new byte[brgImage.length];
		for (int i = 0; i < brgImage.length; i += 3) {
			rgbImage[i] = brgImage[i + 2];
			rgbImage[i + 1] = brgImage[i + 1];
			rgbImage[i + 2] = brgImage[i];
		}
		return rgbImage;
	}

}
