using UnityEngine;
using UnityEngine.SceneManagement;

public class AudioManager : MonoBehaviour
{
    [Header ("----- Audio source -----")]
    [SerializeField] AudioSource musicSource;
    [SerializeField] AudioSource SFXSource;

    [Header ("----- Audio clip -----")]
    public AudioClip backgroundMusic;
    public AudioClip score;
    public AudioClip collision;
    public AudioClip death;
    public AudioClip button;

    private void Start()
    {
        Debug.Log(SceneManager.GetActiveScene().name);
        if (SceneManager.GetActiveScene().name.Equals("Game"))
        {
            musicSource.clip = backgroundMusic;
            musicSource.Play();
        }
    }

    public void PlaySFX(AudioClip clip)
    {
        SFXSource.PlayOneShot(clip);
    }
}
