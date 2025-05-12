using TMPro;
using UnityEngine;

public class SetScore : MonoBehaviour
{
    public TextMeshProUGUI scoreText;
    public int scoreValue = 0;

    // Update is called once per frame
    void Update()
    {
        scoreText.text = scoreValue.ToString();
    }
}
