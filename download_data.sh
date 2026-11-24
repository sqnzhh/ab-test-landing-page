#!/usr/bin/env bash
# Downloads the public landing-page A/B test dataset (Udacity) into ./data
mkdir -p data && cd data
B=https://raw.githubusercontent.com/nirupamaprv/Analyze-AB-test-Results/master
curl -sSL -O "$B/ab_data.csv" && curl -sSL -O "$B/countries.csv"
