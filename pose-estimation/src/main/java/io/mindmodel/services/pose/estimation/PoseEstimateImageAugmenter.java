/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.mindmodel.services.pose.estimation;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.List;
import java.util.function.BiFunction;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.GraphicsUtils;
import io.mindmodel.services.pose.estimation.domain.Body;
import io.mindmodel.services.pose.estimation.domain.Limb;
import io.mindmodel.services.pose.estimation.domain.Model;
import io.mindmodel.services.pose.estimation.domain.Part;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.springframework.util.CollectionUtils;

/**
 * Extends the {@link } with ability to to augment the input image with the
 * recognized poses.
 *
 * @author Christian Tzolov
 */
public class PoseEstimateImageAugmenter implements BiFunction<byte[], List<Body>, byte[]> {

	private static final Log logger = LogFactory.getLog(PoseEstimateImageAugmenter.class);

	public static final Color DEFAULT_COLOR = new Color(167, 252, 0);

	public enum BodyDrawingColorSchema {monochrome, bodyInstance, limbType}

	private String imageFormat = "jpg";

	/**
	 * When drawPoses is enabled, defines the radius of the oval drawn for each part instance
	 */
	private int drawPartRadius = 4;

	/**
	 * When drawPoses is enabled, defines the line width for drawing the limbs
	 */
	private int drawLineWidth = 2;

	/**
	 * if drawPoses is enabled, drawPartLabels will show the party type ids and description.
	 */
	private boolean drawPartLabels = false;

	/**
	 * When drawPoses is enabled, one can decide to draw all body poses in one color (monochrome), have every
	 * body pose drawn in an unique color (bodyInstance) or use common color schema drawing different limbs.
	 */
	private BodyDrawingColorSchema bodyDrawingColorSchema = BodyDrawingColorSchema.limbType;

	public String getImageFormat() {
		return imageFormat;
	}

	public void setImageFormat(String imageFormat) {
		this.imageFormat = imageFormat;
	}

	public int getDrawPartRadius() {
		return drawPartRadius;
	}

	public void setDrawPartRadius(int drawPartRadius) {
		this.drawPartRadius = drawPartRadius;
	}

	public int getDrawLineWidth() {
		return drawLineWidth;
	}

	public void setDrawLineWidth(int drawLineWidth) {
		this.drawLineWidth = drawLineWidth;
	}

	public boolean isDrawPartLabels() {
		return drawPartLabels;
	}

	public void setDrawPartLabels(boolean drawPartLabels) {
		this.drawPartLabels = drawPartLabels;
	}

	public BodyDrawingColorSchema getBodyDrawingColorSchema() {
		return bodyDrawingColorSchema;
	}

	public void setBodyDrawingColorSchema(BodyDrawingColorSchema bodyDrawingColorSchema) {
		this.bodyDrawingColorSchema = bodyDrawingColorSchema;
	}

	@Override
	public byte[] apply(byte[] inputImage, List<Body> bodies) {
		if (!CollectionUtils.isEmpty(bodies)) {
			try {
				return drawPoses(inputImage, bodies);
			}
			catch (IOException e) {
				logger.error("Failed to draw the poses", e);
			}
		}
		return inputImage;
	}

	private byte[] drawPoses(byte[] imageBytes, List<Body> bodies) throws IOException {

		BufferedImage originalImage = ImageIO.read(new ByteArrayInputStream(imageBytes));

		Graphics2D g = originalImage.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		Stroke stroke = g.getStroke();
		g.setStroke(new BasicStroke(this.getDrawLineWidth()));

		for (Body body : bodies) {
			for (Limb limb : body.getLimbs()) {

				Color limbColor = findLimbColor(body, limb);

				Part from = limb.getFromPart();
				Part to = limb.getToPart();

				if (limb.getLimbType() != Model.LimbType.limb17 && limb.getLimbType() != Model.LimbType.limb18) {
					g.setColor(limbColor);
					g.draw(new Line2D.Double(from.getNormalizedX(), from.getNormalizedY(),
							to.getNormalizedX(), to.getNormalizedY()));
				}

				g.setStroke(new BasicStroke(1));
				drawPartOval(from, this.getDrawPartRadius(), g);
				drawPartOval(to, this.getDrawPartRadius(), g);
				g.setStroke(new BasicStroke(this.getDrawLineWidth()));
			}
		}

		g.setStroke(stroke);
		imageBytes = GraphicsUtils.toImageByteArray(originalImage, this.getImageFormat());
		g.dispose();

		return imageBytes;
	}

	private Color findLimbColor(Body body, Limb limb) {
		Color limbColor = DEFAULT_COLOR;
		switch (this.getBodyDrawingColorSchema()) {
		case bodyInstance:
			limbColor = GraphicsUtils.getClassColor(body.getBodyId() * 3);
			break;
		case limbType:
			limbColor = GraphicsUtils.LIMBS_COLORS[limb.getLimbType().getId()];
			break;
		case monochrome:
			limbColor = DEFAULT_COLOR;
			break;
		}

		return limbColor;
	}

	private void drawPartOval(Part part, int radius, Graphics2D g) {
		int partX = part.getNormalizedX();
		int partY = part.getNormalizedY();

		g.setColor(GraphicsUtils.LIMBS_COLORS[part.getPartType().getId()]);
		g.fillOval(partX - radius, partY - radius, 2 * radius, 2 * radius);

		if (this.isDrawPartLabels()) {
			String label = part.getPartType().getId() + ":" + part.getPartType().name();
			FontMetrics fm = g.getFontMetrics();
			int labelX = partX + 5;
			int labelY = partY - 5;
			AffineTransform t = g.getTransform();
			g.setTransform(AffineTransform.getRotateInstance(Math.toRadians(-35), labelX, labelY));

			g.drawString(label, labelX, labelY);
			g.setTransform(t);
		}

	}
}
