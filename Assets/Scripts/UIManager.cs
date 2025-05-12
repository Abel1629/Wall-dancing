using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

// it handles the button presses, and loads the different scenes with a transition between them

public class UIManager : MonoBehaviour
{
    public void OnStartPress()
    {
        NextScene("Game");
    }

    public void OnQuitPress()
    {
        Application.Quit();
    }

    public void OnRestartPress()
    {
        NextScene("Game");
    }

    public void OnQuitToMenuPress()
    {
        NextScene("Home Screen");
    }

    public void GameOver()
    {
        Invoke("LoadGameOverScene", 3f);
    }

    public void LoadGameOverScene()
    {
        NextScene("Game over");
    }

    public void NextScene(string sceneName)
    {
        StartCoroutine(LoadScene(sceneName));
    }

    IEnumerator LoadScene(string sceneName)
    {
        Debug.Log("Loading scene " + sceneName);
        yield return new WaitForSeconds(0.5f);
        SceneManager.LoadScene(sceneName);
    }
}
