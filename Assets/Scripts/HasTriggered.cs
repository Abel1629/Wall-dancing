using UnityEngine;

// It makes sure, that every wall makes just one new wall
public class HasTriggered : MonoBehaviour
{
    public bool triggered = false;

    public void ChangeTriggerState()
    {
        triggered = true;
    }
}
