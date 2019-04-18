package io.mindmodel.services.image.recognition;

/**
 * @author Christian Tzolov
 */
public class RecognitionResponse {
	private String label;
	private Double probability;

	public RecognitionResponse() {
	}

	public RecognitionResponse(String label, Double probability) {
		this.label = label;
		this.probability = probability;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	public Double getProbability() {
		return probability;
	}

	public void setProbability(Double probability) {
		this.probability = probability;
	}

	@Override
	public String toString() {
		return "{label='" + label + ", probability=" + probability + '}';
	}
}
