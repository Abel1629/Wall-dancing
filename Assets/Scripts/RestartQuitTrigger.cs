using UnityEngine;
using UnityEngine.UI;

public class RestartQuitTrigger : MonoBehaviour
{
    [Header("--- Set the player positions ---")]
    [SerializeField] private GameObject playerT;
    [SerializeField] private GameObject playerX;

    [Header("--- Set the UIManager and the images ---")]
    [SerializeField] private UIManager uimanager;
    [SerializeField] private Image imageGreen;
    [SerializeField] private Image imageRed;

    private Color imageGreenColor;
    private Color imageRedColor;

    private float activeTimeGreen = 0f;
    private float activeTimeRed = 0f;
    private bool actionTriggered = false;

    private void Start()
    {
        imageGreenColor = imageGreen.color;
        imageRedColor = imageRed.color;
    }


    void Update()
    {
        // change transparency and call method for start
        if (playerT.activeInHierarchy) // if the GameObject is active, the timer starts
        {
            activeTimeGreen += Time.deltaTime;

            imageGreenColor.a = Mathf.Clamp01(activeTimeGreen / 3f); // fade in as it nears 3 seconds
            imageGreen.color = imageGreenColor;

            if (!actionTriggered && activeTimeGreen >= 3f) // if the timer reaches three seconds
            {
                actionTriggered = true;

                uimanager.OnRestartPress();
            }
        }
        else
        {
            // Reset timer if object is not active
            activeTimeGreen = 0f;
            actionTriggered = false;

            // Reset the pictures transparency
            imageGreenColor.a = 0f;
            imageGreen.color = imageGreenColor;
        }

        // change transparency and call method for quit
        if (playerX.activeInHierarchy) // if the GameObject is active, the timer starts
        {
            activeTimeRed += Time.deltaTime;

            imageRedColor.a = Mathf.Clamp01(activeTimeRed / 3f); // fade in as it nears 3 seconds
            imageRed.color = imageRedColor;

            if (!actionTriggered && activeTimeRed >= 3f) // if the timer reaches three seconds
            {
                actionTriggered = true;

                uimanager.OnQuitToMenuPress();
            }
        }
        else
        {
            // Reset timer if object is not active
            activeTimeRed = 0f;
            actionTriggered = false;

            // Reset the pictures transparency
            imageRedColor.a = 0f;
            imageRed.color = imageRedColor;
        }
    }
}
